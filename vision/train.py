"""Train the circular convolutional model on CIFAR-10, CIFAR-100 or ImageNet32 with EP.

Every layer has its own value node, the 3-convolution encoder included, so every weight is
updated from the prediction error of the node it projects to and the activity of the
node it reads from. Value nodes z = (h1, h2, h3, u, s) with h0 := x:

    h_l_hat = layer_l(h_{l-1})     (layer_l = one step of vision.model.embed:
                                    conv, then relu + affine GroupNorm; the
                                    last layer is linear)
    u_hat(s), s_hat(u, h3)         (the circular block of vision.model, x_inj := h3)
    E = pi/2 sum_l ||h_l - h_l_hat||^2 + 1/2||u - u_hat(s)||^2 + 1/2||s - s_hat(u, h3)||^2

with pi the precision of the encoder energies (--stem_prec, default 1).
Free phase: h = feedforward encoder (the intermediates of vision.model.embed) followed by
core.ep.free_phase on (u, s) with x_inj = h3.
Nudged phase: coupled Gauss-Seidel sweeps on (s, e) with e over all value nodes (the
coupled step of core.ep extended to the encoder), optionally damped; the encoder errors
are precision weighted, e_l = pi (h_l - h_l_hat):
    forward   h_l = h_l_hat(h_{l-1}) + e_l / pi;  u = u_hat(s) + e_u;  s' = s_hat(u, h3) + e_s
    backward  e_u' = VJP_u[s_hat(., h3)](e_s) - beta gC_u
              e_s' = VJP_s'[u_hat](e_u')     - beta gC_s
              e_3' = VJP_h3[s_hat(u, .)](e_s')
              e_l' = VJP_h_l[h_{l+1}_hat](e_{l+1}')      (l = 2, 1)
Weight gradient: -VJP_theta[th -> (h_l_hat, u_hat, s_hat)](e) / (beta_signed * B) at the
nudged value nodes, with a random sign of beta at every step; each encoder layer sees only
its own error and its input value node.

Presets (beta = beta_scale * recipe beta); the defaults are the settings of the reported
results:
    cifar100    CIFAR recipe (vision.recipes, 180 epochs, batch 64), plain sweeps
                (damp 1.0, cT 20), beta_scale 1.0
    cifar10     CIFAR recipe on CIFAR-10 data, damped sweeps (damp 0.3, cT 40),
                beta_scale 1.0
    imagenet32  ImageNet32 recipe (vision.imagenet32.recipe, 50 epochs, batch 128), plain
                sweeps (damp 1.0, cT 10), beta_scale 0.5; the training set stays in host
                memory (cfg["batch_fn"]) and the selection metric is the plain 40-sweep
                evaluation
--damp / --cT / --beta_scale override the preset.

    python -m vision.train --dataset cifar10|cifar100|imagenet32 [--seed S] [--epochs N]
                           [--damp D] [--cT T] [--stem_prec P] [--beta_scale B]
                           [--data_dir DIR] [--out DIR] [--ckpt_dir DIR] [--resume]
                           [--smoke]

Outputs: <out>/<tag>.json (history, best accuracy), <ckpt_dir>/<tag>_best.pkl (best
parameters) and <ckpt_dir>/<tag>_ckpt.pkl (resume state); --out defaults to
runs/vision/<dataset>/s<seed> and --ckpt_dir to --out. tag = <dataset>_s<seed>, plus
_prec<P> when --stem_prec != 1 and _beta<B> when --beta_scale differs from the preset's.

CIFAR is read from (or downloaded to) data/cifar; ImageNet32 reads the cache built by
`python -m vision.imagenet32.prepare` (data/imagenet32 or $IMAGENET32_ROOT).
"""
import argparse
import functools
import json
import math
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from core import ep as E
from core import train as T
from vision import data as D
from vision import model as M
from vision.recipes import RECIPE, build_model, build_cfg
from vision.imagenet32 import recipe as IR
from vision.imagenet32 import data as ID

_tmap = jax.tree_util.tree_map

# dataset -> (nudge damping, nudge sweeps, beta multiplier)
PRESETS = {
    "cifar100": dict(damp=1.0, cT=20, beta_scale=1.0),
    "cifar10": dict(damp=0.3, cT=40, beta_scale=1.0),
    "imagenet32": dict(damp=1.0, cT=10, beta_scale=0.5),
}

# Defaults of the training configuration read by `run` and `make_train_step`; the
# recipes' build_cfg supplies the rest (epochs, bs, lr, wd, beta, mode="random",
# form="coupled", free / nudge / eval_free / eval_free_aa free-phase configurations of
# core.ep.free_phase, var_pen, var_target, cap, ...).
#   eval_solver   "both": select on the Anderson evaluation (eval_free_aa) and log the
#                 plain one (eval_free); "plain": the other way round
#   val_split     fraction of the training set held out for selection; 0.0 selects on
#                 the test set
#   aug_fn        callable(key, x) -> x applied to every training batch (on device)
#   batch_fn      callable(key, idx) -> (x, y) building every training batch outside jit
#                 (replaces the on-device gather and aug_fn; ImageNet32)
#   resume        continue from <ckpt_dir>/<tag>_ckpt.pkl if it exists
DEFAULTS = dict(
    lr_cycles=1, lr_floor_frac=0.0, seed=0, form="coupled",
    eval_solver="plain", var_pen=0.0, var_target=0.25, conv_tol=0.0, cap=0.0,
    val_split=0.0, report_every=5, aug_fn=None, batch_fn=None, resume=False,
)


# ============================ encoder value nodes ============================
def stem_fns(theta):
    """Per-layer prediction closures fns[l](h_{l-1}) = h_l_hat (h_{-1} := x)."""
    pads = M.stem_pads(len(theta["stem"]))
    return tuple(functools.partial(M.stem_layer, layer, M.STEM_STRIDES[i], pads[i])
                 for i, layer in enumerate(theta["stem"]))


def stem_preds(theta, x, hs):
    """Predictions (h1_hat(x), h2_hat(h1), h3_hat(h2), ...) from the given value nodes."""
    fns = stem_fns(theta)
    assert len(hs) == len(fns), (len(hs), len(fns))
    ins = (x,) + tuple(hs[:-1])
    return tuple(f(h) for f, h in zip(fns, ins))


def stem_forward(theta, x):
    """Feedforward encoder value nodes (h1, h2, h3); h3 == vision.model.embed(theta, x)."""
    hs, h = [], x
    for f in stem_fns(theta):
        h = f(h)
        hs.append(h)
    return tuple(hs)


# ================================ nudged phase ================================
def gs_step(model, theta, psi, x, y, s, e, beta, stem_prec=1.0):
    """One coupled Gauss-Seidel step on (s, e), e = (e_1..e_L, e_u, e_s).
    Returns (s', e', hs, z) with hs, z = (u, s') the reconstruction at (s, e).
    With e_1..e_L = 0 the (s', e_u', e_s') part is the coupled step of core.ep
    (x_inj = h_L = embed). stem_prec = pi (static float): e_1..e_L are the
    precision-weighted errors, h_l = h_l_hat + e_l / pi."""
    sf = stem_fns(theta)
    L = len(sf)
    assert len(e) == L + 2, (len(e), L)
    e_h, e_u, e_s = e[:L], e[L], e[L + 1]
    hs, prev = [], x
    for l in range(L):
        if stem_prec == 1.0:
            prev = sf[l](prev) + e_h[l]
        else:
            prev = sf[l](prev) + e_h[l] / stem_prec
        hs.append(prev)
    h_top = hs[-1]
    u = model.u_hat(theta, s) + e_u
    s_new = model.s_hat(theta, u, h_top) + e_s
    z = (u, s_new)
    gC = jax.grad(lambda zz: model.cost_sum(psi, zz, y))(z)
    # loop part: e_u from the old e_s ...
    _, vjp_su = jax.vjp(lambda uu: model.s_hat(theta, uu, h_top), u)
    e_u_new = vjp_su(e_s)[0] - E.bcast_batch(beta, gC[0]) * gC[0]
    # ... e_s from the just-updated e_u, Jacobian of u_hat at the rebuilt s'
    _, vjp_us = jax.vjp(lambda ss: model.u_hat(theta, ss), s_new)
    e_s_new = vjp_us(e_u_new)[0] - E.bcast_batch(beta, gC[1]) * gC[1]
    # encoder part, backward from the just-updated e_s
    _, vjp_sh = jax.vjp(lambda hh: model.s_hat(theta, u, hh), h_top)
    e_h_new = [None] * L
    e_h_new[L - 1] = vjp_sh(e_s_new)[0]
    for l in range(L - 2, -1, -1):
        _, vjp_l = jax.vjp(sf[l + 1], hs[l])
        e_h_new[l] = vjp_l(e_h_new[l + 1])[0]
    return s_new, tuple(e_h_new) + (e_u_new, e_s_new), tuple(hs), z


def reconstruct(model, theta, x, s, e, stem_prec=1.0):
    """Final reconstruction as in core.ep: every value node rebuilt from its
    predecessor plus its error, except s (the tracked value); encoder nodes
    h_l = h_l_hat + e_l / stem_prec (see gs_step)."""
    sf = stem_fns(theta)
    L = len(sf)
    hs, prev = [], x
    for l in range(L):
        if stem_prec == 1.0:
            prev = sf[l](prev) + e[l]
        else:
            prev = sf[l](prev) + e[l] / stem_prec
        hs.append(prev)
    u = model.u_hat(theta, s) + e[L]
    return tuple(hs) + (u, s)


def _fro(a):
    return jnp.sqrt(jnp.sum(a * a))


def nudged_phase(model, theta, psi, x, y, hs_free, z_free, beta, T, return_delta=False,
                 damp=1.0, stem_prec=1.0):
    """T coupled sweeps (gs_step) from s = z_free[-1], e = 0.
    damp != 1.0: each sweep is state <- (1 - damp) * state + damp * GS(state) on
    the full (s, e) state; damp == 1.0 is the plain sweep. stem_prec: precision
    of the encoder energies (static). Returns (zv, e) with zv = (h1, .., hL, u, s)
    the nudged value nodes and e = (e_1..e_L, e_u, e_s).
    return_delta=True -> (zv, e, dE), dE[k] = ||e_T - e_{T-1}|| / ||e_T|| per
    value node (whole-batch Frobenius norms; same numerics, the last sweep is
    run outside the fori_loop)."""
    T = int(T)
    assert T >= 1, T
    s0 = z_free[-1]
    e0 = tuple(jnp.zeros_like(h) for h in hs_free) + tuple(jnp.zeros_like(v) for v in z_free)

    def body(_, carry):
        s, e = carry
        s_new, e_new, _hs, _z = gs_step(model, theta, psi, x, y, s, e, beta,
                                        stem_prec=stem_prec)
        if damp != 1.0:
            return _tmap(lambda o, n: (1.0 - damp) * o + damp * n, (s, e), (s_new, e_new))
        return (s_new, e_new)

    if return_delta:
        s, e_prev = jax.lax.fori_loop(0, T - 1, body, (s0, e0))
        s, e = body(T - 1, (s, e_prev))
        dE = jnp.stack([_fro(a - b) / (_fro(a) + 1e-30) for a, b in zip(e, e_prev)])
    else:
        s, e = jax.lax.fori_loop(0, T, body, (s0, e0))
    zv = reconstruct(model, theta, x, s, e, stem_prec=stem_prec)
    return (zv, e, dE) if return_delta else (zv, e)


# ================================ weight gradient ================================
def theta_grad(model, theta, x, hs, z, e, beta_signed):
    """-VJP_theta[th -> (stem_preds(th, x, hs), u_hat(th, s), s_hat(th, u, h_L))](e)
    / (beta_signed * B). h_L is the nudged value node (not embed(th, x)), so each
    encoder layer is trained by its own error and its input value node only."""
    B = x.shape[0]
    hs = tuple(hs)
    u, s = z[0], z[-1]
    h_top = hs[-1]

    def pred(th):
        return stem_preds(th, x, hs) + (model.u_hat(th, s), model.s_hat(th, u, h_top))

    _, vjp = jax.vjp(pred, theta)
    w = 1.0 / (beta_signed * B)
    werr = tuple(E.bcast_batch(w, a) * a for a in e)
    (g,) = vjp(werr)
    return _tmap(lambda t: -t, g)


# ================================== train step ==================================
def make_train_step(model, cfg, optimizer):
    """The jitted training step (random-sign beta, coupled nudged phase): free phase
    (feedforward encoder + core.ep.free_phase on (u, s)), nudged_phase and theta_grad
    above (encoder precision cfg["stem_prec"], default 1.0), the variance penalty
    (x_inj held fixed), the readout gradient of the free-phase cross-entropy and one
    optimiser update. logs: e_free, loss, acc (training batch) and dE (relative error
    change of the last nudge sweep, per value node)."""
    assert cfg["mode"] == "random", cfg["mode"]
    assert cfg.get("form", "coupled") == "coupled", cfg.get("form")
    assert cfg.get("conv_tol", 0.0) == 0.0, "conv_tol gate not supported here"
    nudge_cfg = cfg["nudge"]
    assert nudge_cfg.get("m", 0) < 2 and not nudge_cfg.get("block", False), nudge_cfg
    beta = cfg["beta"]
    free_cfg = cfg["free"]
    T_n = int(nudge_cfg["T"])
    damp = float(nudge_cfg.get("damp", 1.0))
    stem_prec = float(cfg.get("stem_prec", 1.0))
    var_pen, var_target = cfg.get("var_pen", 0.0), cfg.get("var_target", 0.25)

    def step(params, opt_state, x, y, key):
        theta, psi = params["theta"], params["psi"]
        hs_free = stem_forward(theta, x)
        x_inj = hs_free[-1]
        z_free = E.free_phase(model, theta, x_inj, free_cfg)
        L = len(hs_free)

        sign = jnp.where(jax.random.bernoulli(key, 0.5), 1.0, -1.0)
        zv, e_n, dE = nudged_phase(model, theta, psi, x, y, hs_free, z_free, sign * beta,
                                   T_n, return_delta=True, damp=damp, stem_prec=stem_prec)
        th_grad = theta_grad(model, theta, x, zv[:L], zv[L:], e_n, sign * beta)
        B_full = x.shape[0]

        if var_pen > 0.0:
            x_fix = jax.lax.stop_gradient(x_inj)
            pen_grad = jax.grad(
                lambda th: var_pen * model.var_penalty(th, z_free, x_fix, var_target))(theta)
            th_grad = _tmap(lambda g, pg: g + pg, th_grad, pen_grad)

        psi_grad = jax.grad(lambda ps: model.ce_loss(ps, z_free, y))(psi)

        updates, opt_state = optimizer.update(
            {"theta": th_grad, "psi": psi_grad}, opt_state, params)
        params = optax.apply_updates(params, updates)
        logs = {"e_free": model.energy_internal(theta, z_free, x_inj) / B_full,
                "loss": model.ce_loss(psi, z_free, y),
                "acc": model.accuracy(psi, z_free, y),
                "dE": dE}
        return params, opt_state, logs

    return jax.jit(step)


# ===================================== loop =====================================
def node_names(n_stem):
    """Names of the value nodes: h1..h<n_stem>, u, s."""
    return [f"h{i + 1}" for i in range(n_stem)] + ["u", "s"]


def _r(v):
    v = float(v)
    return float(f"{v:.6g}") if math.isfinite(v) else v


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def run(model, params, data, cfg, outdir, ckptdir=None, tag="vision"):
    """Train `model` from `params` on data = (Xtr, Ytr, Xte, Yte) with the configuration
    cfg (merged over DEFAULTS): Adam with decoupled weight decay and gradient clipping
    on the core.train schedule, the spectral projection after every step, and an
    evaluation with both free-phase solvers after every report_every epochs. With
    cfg["eval_solver"] "both" (CIFAR) the selection metric `acc` is the Anderson
    evaluation and `acc_plain` is logged; with "plain" (ImageNet32) `acc` is the plain
    sweep evaluation and `acc_aa` is logged. Also logs the epoch-mean training loss /
    accuracy and the per-node nudge convergence dE. Rewrites <outdir>/<tag>.json at
    every report; <tag>_best.pkl and <tag>_ckpt.pkl go to ckptdir (default outdir).
    Returns the history dict plus the final params and logs."""
    c = {**DEFAULTS, **cfg}
    assert c["eval_solver"] in ("both", "plain"), c["eval_solver"]
    plain_sel = c["eval_solver"] == "plain"
    ckptdir = ckptdir or outdir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)

    Xtr, Ytr, Xte, Yte = data
    if c["val_split"] > 0.0:
        n_val = int(round(c["val_split"] * Xtr.shape[0]))
        Xsel, Ysel = Xtr[:n_val], Ytr[:n_val]
        Xtr, Ytr = Xtr[n_val:], Ytr[n_val:]
    else:                                   # select on the test set
        Xsel, Ysel = Xte, Yte
    NTR = Xtr.shape[0]

    sched = T.make_schedule(c["lr"], c["lr_cycles"], max(1, c["epochs"] * (NTR // c["bs"])),
                            c["lr_floor_frac"])
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.add_decayed_weights(c["wd"]), optax.adam(sched))
    opt_state = opt.init(params)
    step_fn = make_train_step(model, c, opt)
    eval_aa = T.make_eval_fn(model, c["eval_free_aa"])
    eval_plain = T.make_eval_fn(model, c["eval_free"])
    acc_add = jax.jit(lambda a, b: _tmap(jnp.add, a, b))
    names = node_names(len(params["theta"]["stem"]))

    proj = (jax.jit(lambda p: model.project_spectral(p, cap=c["cap"]))
            if c.get("cap", 0.0) > 0.0 else (lambda p: p))

    if c.get("batch_fn") is not None:
        get_batch = c["batch_fn"]                   # called outside jit (host-side train set)
    else:
        aug_fn = c["aug_fn"] or (lambda key, x: x)

        @jax.jit
        def get_batch(key, idx):
            return aug_fn(key, Xtr[idx]), Ytr[idx]

    start_ep = 0
    key = jax.random.PRNGKey(c["seed"] + 1)
    hist = []
    best_acc, best_ep, diverged_ep = 0.0, -1, None
    ckpt_path = os.path.join(ckptdir, f"{tag}_ckpt.pkl")
    json_path = os.path.join(outdir, f"{tag}.json")
    if c.get("resume", False) and os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as fh:
            saved = pickle.load(fh)
        params, opt_state = saved["params"], saved["opt_state"]
        key, start_ep = saved["key"], saved["epoch"]
        hist = saved["hist"]
        best_acc, best_ep = saved["best_acc"], saved["best_epoch"]
        print(f"[{tag}] resumed at epoch {start_ep}", flush=True)

    def _out(secs):
        return dict(tag=tag, cfg=c, best_acc=best_acc, best_epoch=best_ep,
                    diverged_ep=diverged_ep, hist=hist,
                    sec_per_epoch=float(np.mean(secs)) if secs else float("nan"))

    logs, secs = None, []
    re_ = c["report_every"]
    nsteps = NTR // c["bs"]
    for ep in range(start_ep, c["epochs"]):
        key, kp = jax.random.split(key)
        perm = jax.random.permutation(kp, NTR)
        t0 = time.time()
        sums = None
        for i in range(nsteps):
            idx = perm[i * c["bs"]:(i + 1) * c["bs"]]
            key, ka, ks = jax.random.split(key, 3)
            x, y = get_batch(ka, idx)
            params, opt_state, logs = step_fn(params, opt_state, x, y, ks)
            params = proj(params)
            cur = (logs["loss"], logs["acc"], logs["dE"])
            sums = cur if sums is None else acc_add(sums, cur)
        jax.block_until_ready(params)
        sec = time.time() - t0
        secs.append(sec)
        if not math.isfinite(float(logs["loss"])):
            diverged_ep = ep + 1
            print(f"[{tag}] diverged: non-finite loss at ep{ep + 1}", flush=True)
            break
        if (ep + 1) % re_ == 0 or ep == 0 or ep + 1 == c["epochs"]:
            if plain_sel:
                acc = T.evaluate(eval_plain, params, Xsel, Ysel)
                acc_aa = T.evaluate(eval_aa, params, Xsel, Ysel)
            else:
                acc = T.evaluate(eval_aa, params, Xsel, Ysel)
                acc_plain = T.evaluate(eval_plain, params, Xsel, Ysel)
            if acc > best_acc:
                best_acc, best_ep = acc, ep + 1
                with open(os.path.join(ckptdir, f"{tag}_best.pkl"), "wb") as fh:
                    pickle.dump({"params": _tmap(np.asarray, params), "config": c,
                                 "acc": acc, "ep": ep + 1}, fh)
            trdE = np.asarray(sums[2]) / nsteps
            row = dict(ep=ep + 1, acc=round(acc, 4), best=round(best_acc, 4),
                       loss=round(float(logs["loss"]), 4),
                       e_free=round(float(logs["e_free"]), 4),
                       lr=float(sched(ep * nsteps)))
            if plain_sel:
                row["acc_aa"] = round(acc_aa, 4)
            else:
                row["acc_plain"] = round(acc_plain, 4)
            row.update(train_loss=round(float(sums[0]) / nsteps, 4),
                       train_acc=round(float(sums[1]) / nsteps, 4),
                       **{f"tr_dE_{nm}": _r(trdE[i]) for i, nm in enumerate(names)},
                       sec=round(sec, 2))
            hist.append(row)
            print(f"EVAL tag={tag} ep={row['ep']} acc={row['acc']} "
                  f"best={row['best']} loss={row['loss']} e_free={row['e_free']}",
                  flush=True)
            diag_keys = [k for k in row if k not in ("ep", "acc", "best", "loss", "e_free")]
            print(f"DIAG tag={tag} ep={row['ep']} "
                  + " ".join(f"{k}={row[k]}" for k in diag_keys), flush=True)
            _write_json(json_path, _out(secs))
            with open(ckpt_path, "wb") as fh:
                pickle.dump({"params": params, "opt_state": opt_state, "key": key,
                             "epoch": ep + 1, "hist": hist, "best_acc": best_acc,
                             "best_epoch": best_ep}, fh)

    out = _out(secs)
    _write_json(json_path, out)
    print(f"[{tag}] DONE best_acc={best_acc:.4f} @ep{best_ep}", flush=True)
    out["params"], out["last_logs"] = params, logs
    return out


# =================================== presets ===================================
def build_preset(dataset, seed, epochs=None, damp=None, cT=None, stem_prec=1.0,
                 beta_scale=None, smoke=False):
    """Preset -> (params, cfg); sets vision.model's trace-time globals. damp / cT /
    beta_scale None = the preset's; beta = beta_scale * recipe beta (beta_scale
    1.0: the recipe's beta, not multiplied); smoke: 1 epoch, batch 8."""
    preset = PRESETS[dataset]
    damp = damp if damp is not None else preset["damp"]
    cT = cT if cT is not None else preset["cT"]
    beta_scale = beta_scale if beta_scale is not None else preset["beta_scale"]
    in32 = dataset == "imagenet32"
    r = dict(IR.RECIPE if in32 else RECIPE)
    if beta_scale != 1.0:
        r["beta"] = r["beta"] * beta_scale
    epochs = epochs if epochs is not None else r["epochs"]
    if in32:
        params = IR.build_model(seed, r)
    else:
        params = build_model(dataset, seed, r)
    if smoke:
        epochs, r["bs"] = 1, 8
    if in32:
        cfg = IR.build_cfg(r, epochs, seed)         # eval_solver "plain" (selection)
        cfg["eval_free_aa"] = dict(solver="anderson", reduced=True, m=r["m"], T=r["free_t"])
    else:
        cfg = build_cfg(r, epochs, seed)
    cfg["nudge"] = {**cfg["nudge"], "T": cT, "damp": damp}
    cfg["stem_prec"] = float(stem_prec)
    return params, cfg


def run_tag(dataset, seed, stem_prec=1.0, beta_scale=None):
    """Run name: <dataset>_s<seed>, plus _prec<P> when stem_prec != 1.0 and
    _beta<B> when beta_scale (None = the preset's) differs from the preset's."""
    tag = f"{dataset}_s{seed}"
    if stem_prec != 1.0:
        tag += f"_prec{stem_prec:g}"
    if beta_scale is not None and beta_scale != PRESETS[dataset]["beta_scale"]:
        tag += f"_beta{beta_scale:g}"
    return tag


def load_imagenet32(root=None, n_train=None, n_val=None):
    """ImageNet32 for the imagenet32 preset: (XTR, YTR, XVA, YVA) with the uint8
    train set on the host and the normalised val set on the device, plus the
    host-gather batch_fn for cfg["batch_fn"] (vision.imagenet32.recipe).
    n_train / n_val keep only the first images (smoke tests)."""
    XTR, YTR, XVA, YVA, MEAN, STD = ID.get_imagenet32(root, device=False)
    if n_train is not None:
        XTR, YTR = XTR[:n_train], YTR[:n_train]
    if n_val is not None:
        XVA, YVA = XVA[:n_val], YVA[:n_val]
    XTR = np.ascontiguousarray(XTR)                 # memmap -> host RAM
    YTR = np.asarray(YTR)
    XVA = jnp.asarray(np.ascontiguousarray(XVA))
    YVA = jnp.asarray(YVA)
    XVA = jax.jit(lambda x: ID.normalize(x, MEAN, STD))(XVA)
    batch_fn = IR.PicklableBatchFn(IR.make_batch_fn(XTR, YTR, MEAN, STD, D.augment))
    return (XTR, YTR, XVA, YVA), batch_fn


# ===================================== CLI =====================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=("cifar10", "cifar100", "imagenet32"),
                    help="dataset preset (see the module docstring)")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed (initialisation, data order, nudge signs)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="training epochs (default: the recipe's, CIFAR 180, ImageNet32 50)")
    ap.add_argument("--damp", type=float, default=None,
                    help="damped nudge sweeps: state <- (1-damp) state + damp GS(state); "
                         "1.0 = plain (default: dataset preset)")
    ap.add_argument("--cT", type=int, default=None,
                    help="coupled nudge sweeps (default: dataset preset)")
    ap.add_argument("--stem_prec", type=float, default=1.0,
                    help="precision pi of the encoder energies pi/2 ||h_l - h_l_hat||^2 "
                         "(default 1.0)")
    ap.add_argument("--beta_scale", type=float, default=None,
                    help="beta = beta_scale * recipe beta (default: dataset preset, "
                         "1.0 for CIFAR, 0.5 for ImageNet32)")
    ap.add_argument("--data_dir", default=None,
                    help="dataset directory (default: data/cifar; ImageNet32: "
                         "data/imagenet32 or $IMAGENET32_ROOT)")
    ap.add_argument("--out", default=None,
                    help="directory for <tag>.json (default: runs/vision/<dataset>/s<seed>)")
    ap.add_argument("--ckpt_dir", default=None,
                    help="directory for <tag>_best.pkl / <tag>_ckpt.pkl (default: --out)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <ckpt_dir>/<tag>_ckpt.pkl if it exists")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: 1 epoch, batch 8, first 16 train / 16 test images")
    args = ap.parse_args()

    params, cfg = build_preset(args.dataset, args.seed, args.epochs, args.damp, args.cT,
                               args.stem_prec, args.beta_scale, args.smoke)
    cfg.update(resume=args.resume)
    tag = run_tag(args.dataset, args.seed, args.stem_prec, args.beta_scale)
    out = args.out or os.path.join("runs", "vision", args.dataset, f"s{args.seed}")
    shown = {k: v for k, v in cfg.items() if k not in ("aug_fn", "batch_fn")}
    print(f"[{tag}] devices={jax.devices()} act={M.ACT} gn_var_floor={M.GN_VAR_FLOOR} "
          f"stem_strides={M.STEM_STRIDES} stem_pads={M.STEM_PADS} cfg={shown}", flush=True)
    n = 16 if args.smoke else None
    if args.dataset == "imagenet32":
        data, cfg["batch_fn"] = load_imagenet32(args.data_dir, n, n)
    else:
        data = (D.get_cifar100 if args.dataset == "cifar100" else D.get_cifar10)(
            args.data_dir or D.DEFAULT_ROOT)
        if args.smoke:
            data = tuple(a[:16] for a in data)
    res = run(M, params, data, cfg, out, args.ckpt_dir, tag=tag)
    print(f"[{tag}] best_acc={res['best_acc']:.4f} @ep{res['best_epoch']} "
          f"sec/ep={res['sec_per_epoch']:.1f}", flush=True)


if __name__ == "__main__":
    main()
