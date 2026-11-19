"""Train the standard Sudoku model (C=512, 250 epochs) with EP from random initialisation.

    python -m sudoku.train [--seed S] [--epochs 250] [--out DIR] [--data_dir DIR]
                           [--beta B] [--nudge_T T] [--nudge_m M]

writes best_canon.pkl / best_eq.pkl (best validation board accuracy of the canonical /
damped readout) and final.pkl to --out (default runs/sudoku/s<seed>); evaluate them with
`python -m sudoku.evaluate --ckpt DIR/best_eq.pkl`. The data (`python -m sudoku.prepare`)
is read from data/sudoku, $SUDOKU_RRN_DIR or --data_dir. Smoke test (CPU):
    python -m sudoku.train --epochs 1 --bs 4 --max_steps 2 --n_valid 8 --out DIR

The recipe (RECIPE):
  * free phase: Anderson acceleration (m=13, T=150, core.solvers.anderson(fast=True)) of
    the main value node (core.ep.free_phase), gated per sample by core.ep.free_phase_mask
    (relative residual < conv_tol = 1e-3);
  * nudged phase: coupled Gauss-Seidel message passing Anderson-accelerated (m=2, T=5)
    on the readout block (core.ep.nudged_phase), once at +beta and once at -beta,
    beta = 0.1657569721347532;
  * weight gradient ("ec" estimator): the mean of the two one-sided EP gradients
    (core.ep.coupled_theta_grad) at +beta and -beta, each weighting a sample by its
    free-phase gate w and normalising by the number of admitted samples max(sum(w), 1):
        g_+ = coupled_theta_grad(theta, x, z_+, e_+, max(sum(w), 1),  w / beta)
        g_- = coupled_theta_grad(theta, x, z_-, e_-, max(sum(w), 1), -w / beta)
        theta_grad = (g_+ + g_-) / 2 + d/dtheta var_pen * var_penalty_pred
  * readout psi: gradient of the free-phase cross-entropy;
  * optimiser: clip -> decoupled weight decay -> Adam on a warmup-cosine schedule whose
    horizon H1 (60 epochs) is shorter than the run, then a floor lr_floor; after every
    step the update of the token- / channel-mixing leaves is scaled by tok_lr_scale /
    chan_lr_scale and the mixing matrices are projected to spectral norm <= cap_eff;
  * spectral cap: once per epoch cap_eff is adjusted by a log-space P-controller towards
    a spectral radius rho(J) = rho_target of the loop Jacobian (rho_update);
  * checkpoints: best_canon.pkl / best_eq.pkl on strict improvement of the validation
    board accuracy (first n_valid validation boards), final.pkl at the end.
"""
import argparse
import math
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from core import diagnostics as D
from core import ep as E
from . import data as Dat
from . import evaluate as Ev
from . import model as M

_tmap = jax.tree_util.tree_map

# ----------------------------------------------------------------- recipe
# The standard Sudoku recipe (C=512).
RECIPE = dict(
    N=9, C=512, r_t=2.0, r_c=2.0, depth=1, inject="all",
    act_spec="gelu@2.35804809692989", norm_eps=0.0020552262184499793,
    gain=0.25, cap=0.7,
    T_free=150, conv_tol=0.001, m_free=13,
    beta=0.1657569721347532, nudge_m=2, nudge_T=5,
    var_pen=2.916625208233585, var_target=0.09866012962332393,
    wd=0.00012144818116425606, adam_b1=0.9,
    lr=5.2e-4, H1=60, warmup=186, lr_floor=6e-5,
    tok_lr_scale=0.9183663599862932, chan_lr_scale=0.4347694630102851,
    rho_target=1.25, rho_kp=0.2, rho_band=0.05, rho_cap_lo=0.30,
    bs=256, ep_size=8000, clip=1.0, augment=1,
)


def init_params(seed, c):
    """M.init_params(PRNGKey(seed), init_scale=gain) -> M.project_spectral(cap).
    M.configure(c) must have been called first."""
    params = M.init_params(jax.random.PRNGKey(seed), C=c["C"], init_scale=c["gain"])
    return M.project_spectral(params, cap=c["cap"])


def make_schedule(c, spe):
    """optax.warmup_cosine_decay_schedule(0, lr, warmup, H1*spe) floored at
    lr_floor -- the cosine horizon H1 (epochs) is independent of the run
    length; epochs beyond H1 train at the floor."""
    total = c["H1"] * spe
    sched = optax.warmup_cosine_decay_schedule(0.0, c["lr"], c["warmup"], total)
    floor = c["lr_floor"]
    return (lambda k: jnp.maximum(sched(k), floor)) if floor > 0.0 else sched


def make_opt(c, sched):
    """clip_by_global_norm(clip) -> add_decayed_weights(wd) -> adam(sched, b1)
    (b2/eps = optax defaults 0.999/1e-8)."""
    tx = [optax.clip_by_global_norm(c["clip"])] if c["clip"] > 0 else []
    tx += [optax.add_decayed_weights(c["wd"]), optax.adam(sched, b1=c["adam_b1"])]
    return optax.chain(*tx)


def make_train_step(c, optimizer):
    """One jitted EP step: gated free phase, nudged phases at +beta / -beta, the "ec"
    weight gradient plus the var_penalty_pred gradient, the readout gradient and one
    optimiser update. The per-layer learning-rate scaling and the spectral projection
    are applied by `run`, outside the step. logs: e_free, loss, gate_frac."""
    beta = c["beta"]
    T_free, m_free = c["T_free"], c["m_free"]
    conv_tol = c["conv_tol"]
    nudge_m, nudge_T = c["nudge_m"], c["nudge_T"]
    var_pen, var_target = c["var_pen"], c["var_target"]

    free_cfg = {"solver": "anderson", "reduced": True, "fast": True,
                "m": m_free, "T": T_free}
    nudge_cfg = {"m": nudge_m, "T": nudge_T}

    def step(params, opt_state, px, py):
        theta, psi = params["theta"], params["psi"]
        x_inj = M.embed(theta, px)
        z_free = E.free_phase(M, theta, x_inj, free_cfg)
        w, _ = E.free_phase_mask(M, theta, z_free, x_inj, conv_tol)
        Bw = jnp.maximum(jnp.sum(w), 1.0)

        zp, ep_ = E.nudged_phase(M, theta, psi, x_inj, py, z_free, beta, nudge_cfg)
        zm, em = E.nudged_phase(M, theta, psi, x_inj, py, z_free, -beta, nudge_cfg)

        g_ep = E.coupled_theta_grad(M, theta, px, zp, ep_, Bw, w / beta)
        g_em = E.coupled_theta_grad(M, theta, px, zm, em, Bw, -w / beta)
        theta_grad = _tmap(lambda a, b: 0.5 * (a + b), g_ep, g_em)

        pen_grad = jax.grad(
            lambda th: var_pen * M.var_penalty_pred(th, z_free, x_inj, var_target))(theta)
        theta_grad = _tmap(lambda a, b: a + b, theta_grad, pen_grad)
        psi_grad = jax.grad(lambda ps: M.ce_loss(ps, z_free, py))(psi)

        updates, opt_state = optimizer.update(
            {"theta": theta_grad, "psi": psi_grad}, opt_state, params)
        params = optax.apply_updates(params, updates)
        logs = dict(e_free=M.energy_internal(theta, z_free, x_inj) / px.shape[0],
                    loss=M.ce_loss(psi, z_free, py), gate_frac=jnp.mean(w))
        return params, opt_state, logs

    return jax.jit(step)


def make_lr_blend(c):
    """Post-Adam per-layer LR-scale blend: new_leaf = old_leaf + scale *
    (new_leaf - old_leaf), for the token-mix leaves (tok_lr_scale) and the
    channel-mix leaves (chan_lr_scale), applied OUTSIDE the jitted train step
    (needs both the pre- and post-step params)."""
    tok_scale, chan_scale = float(c["tok_lr_scale"]), float(c["chan_lr_scale"])

    @jax.jit
    def blend(params, prev_params):
        th, pth = dict(params["theta"]), prev_params["theta"]
        for scale, leaves in ((tok_scale, M.TOK_LEAVES), (chan_scale, M.CHAN_LEAVES)):
            if scale != 1.0:
                for k in leaves:
                    th[k] = pth[k] + scale * (th[k] - pth[k])
        return {**params, "theta": th}

    return blend


def rho_update(cap_eff, cr_now, c):
    """Per-epoch symmetric log-space P-controller on the spectral cap:
    deadband |err| <= rho_band, else cap_eff *= exp(rho_kp * (target - cr_now)),
    clipped to [rho_cap_lo, cap]. A pure function (cap_eff carried by the caller)."""
    err = c["rho_target"] - cr_now
    if abs(err) <= c["rho_band"]:
        return cap_eff
    return float(np.clip(cap_eff * math.exp(c["rho_kp"] * err), c["rho_cap_lo"], c["cap"]))


def to_np(params):
    """The params pytree as numpy arrays (for pickling)."""
    return _tmap(np.asarray, params)


def _save(path, blob):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(blob, fh)
    os.replace(tmp, path)


def run(seed=0, epochs=250, out_dir=None, c=None, tag=None, report_every=1,
        max_steps=None, n_valid=2000, data_dir=None):
    """Train the standard recipe from random init. `out_dir` (if given): writes
    `best_canon.pkl` / `best_eq.pkl` (strict board improvement on the first `n_valid`
    validation boards, ties keep the earlier epoch, epoch 1 always saves both) and
    `final.pkl` (always, at the end or at the first non-finite loss). Returns a dict
    with keys params, hist, best (dict of (board, epoch) per readout), cfg.
    `max_steps` (smoke tests only) stops every epoch after that many steps; `data_dir`
    defaults to sudoku.data.RRN_DIR."""
    c = {**RECIPE, **(c or {})}
    tag = tag or f"sudoku_s{seed}"
    M.configure(c)
    d = Dat.get_sudoku_rrn(split_test="valid", n_test=0, data_dir=data_dir)
    PTR, STR = jnp.asarray(d[0]), jnp.asarray(d[1])
    YTR = STR - 1
    PA, YA = jnp.asarray(d[2][:n_valid]), jnp.asarray(d[3][:n_valid]) - 1
    NTR, BS = PTR.shape[0], c["bs"]
    EP_N = min(NTR, c["ep_size"])
    spe = EP_N // BS

    params = init_params(seed, c)
    sched = make_schedule(c, spe)
    opt = make_opt(c, sched)
    opt_state = opt.init(params)
    step_fn = make_train_step(c, opt)
    lr_blend = make_lr_blend(c)
    proj = jax.jit(lambda p, cap: M.project_spectral(p, cap=cap))
    rho_free_cfg = {"solver": "sweep", "T": 80}

    cap_eff = float(c["cap"])
    key = jax.random.PRNGKey(seed + 1)
    best = {"canon": (-1.0, None), "eq": (-1.0, None)}
    hist, t_train = [], 0.0
    logs = None
    for ep in range(epochs):
        key, kp = jax.random.split(key)
        perm = jax.random.permutation(kp, NTR)
        gate = []
        t0 = time.time()
        for i in range(spe if max_steps is None else min(spe, max_steps)):
            idx = perm[i * BS:(i + 1) * BS]
            key, ka = jax.random.split(key)
            if c["augment"]:
                px, sx = Dat.augment_batch(ka, PTR[idx], STR[idx], c["N"])
                py = sx - 1
            else:
                px, py = PTR[idx], YTR[idx]
            prev_params = params
            params, opt_state, logs = step_fn(params, opt_state, px, py)
            params = lr_blend(params, prev_params)
            params = proj(params, cap_eff)
            gate.append(float(logs["gate_frac"]))
        t_train += time.time() - t0
        if not math.isfinite(float(logs["loss"])):
            print(f"[{tag}] ep{ep + 1} NON-FINITE loss -> stop", flush=True)
            hist.append(dict(ep=ep + 1, nonfinite=True))
            break
        cr_now = D.spectral_radius(M, params["theta"], PA[:256], rho_free_cfg)
        cap_eff = rho_update(cap_eff, cr_now, c)
        params = proj(params, cap_eff)
        cell, board = Ev.canon_eval(params, PA, YA)
        eq_cell, eq_board = Ev.damped_eval(params, PA, YA)
        row = dict(ep=ep + 1, cell=round(cell, 4), board=round(board, 4),
                   eq_cell=round(eq_cell, 4), eq_board=round(eq_board, 4),
                   gate=round(float(np.mean(gate)), 3),
                   e_free=round(float(logs["e_free"]), 4), loss=round(float(logs["loss"]), 4),
                   cr=round(cr_now, 4), cap_eff=round(cap_eff, 4),
                   lr=float(sched(ep * spe)), t_train=round(t_train, 1))
        e1 = ep + 1
        if out_dir is not None:
            for kind, val in (("canon", board), ("eq", eq_board)):
                if val > best[kind][0]:
                    best[kind] = (val, e1)
                    _save(os.path.join(out_dir, f"best_{kind}.pkl"),
                          dict(params=to_np(params), ep=e1, which=kind, board_canon=board,
                               board_eq=eq_board, cell_canon=cell, cell_eq=eq_cell,
                               cap_eff=cap_eff, cr=cr_now, cfg=c))
        hist.append(row)
        if (e1 % report_every == 0) or e1 == 1 or e1 == epochs:
            print(f"[{tag}] {row}", flush=True)
    if out_dir is not None:
        _save(os.path.join(out_dir, "final.pkl"),
              dict(params=to_np(params), ep=len(hist), cap_eff=cap_eff, cfg=c, hist=hist))
    print(f"[{tag}] DONE best_canon={best['canon']} best_eq={best['eq']}", flush=True)
    return dict(params=params, hist=hist, best=best, cfg=c)


def main():
    ap = argparse.ArgumentParser(description="Train the standard Sudoku model "
                                             "(sudoku.train.RECIPE) with EP.")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed (initialisation, data order, augmentation)")
    ap.add_argument("--epochs", type=int, default=250, help="training epochs")
    ap.add_argument("--out", default=None,
                    help="output directory (default: runs/sudoku/s<seed>)")
    ap.add_argument("--data_dir", default=Dat.RRN_DIR,
                    help="directory with train.npz / valid.npz written by sudoku.prepare "
                         "(default: data/sudoku in the repository, or $SUDOKU_RRN_DIR)")
    ap.add_argument("--beta", type=float, default=None,
                    help="override the recipe's nudge strength beta")
    ap.add_argument("--nudge_T", type=int, default=None,
                    help="override the recipe's nudged-phase budget nudge_T")
    ap.add_argument("--nudge_m", type=int, default=None,
                    help="override the recipe's nudged-phase Anderson depth nudge_m (>= 2)")
    ap.add_argument("--bs", type=int, default=None,
                    help="override the recipe's batch size (smoke tests)")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="stop every epoch after this many steps (smoke tests)")
    ap.add_argument("--n_valid", type=int, default=2000,
                    help="validation boards used for checkpoint selection")
    args = ap.parse_args()
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")

    out = args.out or os.path.join("runs", "sudoku", f"s{args.seed}")
    c = dict(RECIPE)
    over = {k: getattr(args, k) for k in ("beta", "nudge_T", "nudge_m", "bs")}
    over = {k: v for k, v in over.items() if v is not None}
    if over:
        c.update(over)
        print(f"recipe overrides: {over}", flush=True)
    res = run(seed=args.seed, epochs=args.epochs, out_dir=out, c=c,
              tag=f"sudoku_s{args.seed}", max_steps=args.max_steps, n_valid=args.n_valid,
              data_dir=args.data_dir)
    print(f"best_canon={res['best']['canon']} best_eq={res['best']['eq']}", flush=True)


if __name__ == "__main__":
    main()
