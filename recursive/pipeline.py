"""The recipe (DEFAULTS), config handling, the training loop and the free-chain evaluator.

One training step carries the answer state y and the ring state z across a
random number of deep-supervision loops per batch (n_sup_min..n_sup_max,
drawn uniformly), each loop being ONE supervised energy minimisation
(ep.stage_grads) followed by one optimiser step (loop_step):

    y <- 0, z <- 0
    for n in 1..n_loops:                         # one optimiser step per loop
        x_inj = embed(puz, y)
        z <- free relaxation of E(.; x_inj)        (cold: from 0, T_free Anderson
                                                     iterations; warm: from z,
                                                     T_free_warm iterations)
        z+/z- <- the fused nudged pair of E +/- beta C from z
        theta_grad = (g(z+) - g(z-)) / 2 beta       (+ the var-penalty term)
        psi_grad   = d/dpsi CE(readout(z))          (delta rule)
        y <- z[-1]                                   # the answer update
    theta, psi <- the optimiser update on this loop's gradients

Evaluation (`chain` / `eval_chain`): the free chain only (no label, no nudge),
the answer state carried exactly as in training; the bank (`train`'s
`best.pkl`) is keyed on the last stage's board accuracy.

Data loading, augmentation and label conventions are NOT this module's
concern: `train` takes puzzle / solution arrays directly and an optional
`augment_fn(key, puzzles, solutions) -> (puzzles, solutions)`.
"""
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

from recursive import ep as EP
from recursive import model as M
from recursive import optimizers as OPT
from recursive import solvers as S

MATMUL_PRECISIONS = {"default": "tensorfloat32", "highest": "highest"}

# The recipe of the recursive model (recursive.train trains it; `--set KEY=VALUE`
# overrides a key) and the complete set of configuration keys of the training and
# evaluation paths.
DEFAULTS = dict(
    # geometry
    N=9, C=120, r_t=2.0, r_c=2.0,
    act_spec="gelu@2.35804809692989", norm_eps=0.0020552262184499793,
    gain=0.25, cap=0.7,
    # EP dynamics
    beta=0.02, T_nudg=500, lr_h=0.02082750120349555,
    mom=0.9917259505704753, m=13, T_free=600, T_free_warm=80,
    var_pen=2.916625208233585, var_target=0.144,
    # deep supervision (loops per batch ~ U[n_sup_min, n_sup_max])
    n_sup_min=4, n_sup_max=16,
    # pruning ("" = off; comma list of nudged/contrast row counts per loop index)
    prune_sched="256,256,128,128,64,64,64,64", prune_keep=0.25,
    # the answer update and per-leaf optimiser lr scaling
    y_rate=1.0, tok_lr_scale=0.9183663599862932, chan_lr_scale=0.4347694630102851,
    # outer optimisation (ep_size: puzzles per epoch, 0 = the whole training set)
    seed=0, bs=128, ep_size=8000, epochs=90, lr=0.0003, lr_min=1e-06, warmup=100,
    wd=0.00012144818116425606, ema=0.999, matmul_precision="highest",
    # evaluation / reporting (n_valid: leading rows of the valid set used by the
    # periodic eval inside train(); 0 = all of it)
    eval_stages=16, report_every=2, n_valid=2000,
)


def merge_cfg(cfg):
    """{**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS}}: missing keys
    take the recipe's value; keys that are not configuration keys are ignored."""
    cfg = cfg or {}
    return {**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS}}


def set_matmul_precision(cfg):
    """Set jax_default_matmul_precision from cfg["matmul_precision"] ("default" =
    TF32, "highest" = true fp32). Call before any jitted computation."""
    jax.config.update("jax_default_matmul_precision",
                      MATMUL_PRECISIONS[cfg["matmul_precision"]])


def parse_prune_sched(s):
    """"" -> (); "256,256,128" -> (256, 256, 128); every entry a positive int."""
    if s is None or str(s).strip() == "":
        return ()
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v <= 0:
            raise ValueError(f"prune_sched entry {v}: must be positive")
        out.append(v)
    return tuple(out)


def prune_rows_for_loop(sched, n, bs):
    """K rows for loop index n of a batch of bs rows: 0 (no pruning) for an empty
    schedule, loop 0 (no previous answer yet) and an entry >= bs; loops beyond
    the schedule reuse its last entry."""
    if not sched or n == 0:
        return 0
    k = int(sched[min(n, len(sched) - 1)])
    return 0 if k >= bs else k


def draw_n_loops(cfg, rng):
    """Loops for one batch, drawn uniformly from [n_sup_min, n_sup_max] with the
    numpy Generator `rng`."""
    return int(rng.integers(int(cfg["n_sup_min"]), int(cfg["n_sup_max"]) + 1))


# ------------------------------ one optimiser step ------------------------------
@functools.partial(jax.jit, static_argnames=(
    "opt", "cold", "prune_k", "T_free", "T_free_warm", "T_nudg", "m", "lr_h", "mom",
    "beta", "var_pen", "var_target", "prune_keep", "tok_lr_scale", "chan_lr_scale",
    "y_rate", "cap"))
def _loop_step_jit(params, opt_state, x, y_lab, y_carry, z_warm, key, *, opt, cold,
                   prune_k, T_free, T_free_warm, T_nudg, m, lr_h, mom, beta, var_pen,
                   var_target, prune_keep, tok_lr_scale, chan_lr_scale, y_rate, cap):
    ep_cfg = dict(T_free=T_free, T_free_warm=T_free_warm, T_nudg=T_nudg, m=m, lr_h=lr_h,
                 mom=mom, beta=beta, var_pen=var_pen, var_target=var_target,
                 prune_keep=prune_keep, y_rate=y_rate)
    grads, z_free, y_new, logs = EP.stage_grads(params, x, y_lab, y_carry, z_warm, key,
                                                cold=cold, prune_k=prune_k, cfg=ep_cfg)
    opt_cfg = dict(tok_lr_scale=tok_lr_scale, chan_lr_scale=chan_lr_scale, cap=cap)
    new_params, new_opt_state = OPT.apply_update(opt, params, grads, opt_state, opt_cfg)
    logs["gnorm_theta"] = optax.global_norm(grads["theta"])
    return new_params, new_opt_state, y_new, z_free, logs


def loop_step(params, opt, opt_state, x, y_lab, y_carry, z_warm, key, *, cold, prune_k, cfg):
    """One optimiser step: ep.stage_grads (one energy minimisation per loop,
    always contrasted) -> optimizers.apply_update. prune_k: 0 (or >= B) = the
    nudged phase + contrast on the whole batch; 0 < prune_k < B = on prune_k
    rows chosen from the previous loop's answer. Every cfg scalar is passed to
    the jitted step as a static (hashable) argument. Returns (params, opt_state,
    y_carry, z_warm, logs); logs are those of ep.stage_grads plus gnorm_theta, the
    global norm of the weight gradient."""
    return _loop_step_jit(
        params, opt_state, x, y_lab, y_carry, z_warm, key, opt=opt, cold=cold,
        prune_k=prune_k, T_free=cfg["T_free"], T_free_warm=cfg["T_free_warm"],
        T_nudg=cfg["T_nudg"], m=cfg["m"], lr_h=cfg["lr_h"], mom=cfg["mom"],
        beta=cfg["beta"], var_pen=cfg["var_pen"], var_target=cfg["var_target"],
        prune_keep=cfg["prune_keep"], tok_lr_scale=cfg["tok_lr_scale"],
        chan_lr_scale=cfg["chan_lr_scale"], y_rate=cfg["y_rate"], cap=cfg["cap"])


# ------------------------------ evaluation ------------------------------
@functools.partial(jax.jit, static_argnames=("stages", "m", "T_free", "T_free_warm", "y_rate"))
def _chain_jit(params, puzzles, *, stages, m, T_free, T_free_warm, y_rate):
    theta, psi = params["theta"], params["psi"]
    B, C = puzzles.shape[0], theta["Ws"].shape[1]
    free_cfg = dict(m=m, T_free=T_free, T_free_warm=T_free_warm)

    def cost_batch(s, yhat):
        """Batch SUM of the per-sample mean-over-cells CE of readout(psi, s) vs
        yhat (samples independent, so d/ds of this sum is the per-sample
        gradient)."""
        logp = jax.nn.log_softmax(M.readout(psi, s), axis=-1)
        nll = -jnp.take_along_axis(logp, yhat[..., None], axis=-1)[..., 0]
        return jnp.sum(jnp.mean(nll, axis=-1))

    def one(y, z, cold):
        x_inj = M.embed(theta, puzzles, y)
        z = S.relax_free(theta, x_inj, z, cold=cold, cfg=free_cfg)
        s = z[-1]
        logits = M.readout(psi, s)
        logp = jax.nn.log_softmax(logits, axis=-1)
        probs = jax.nn.softmax(logits, axis=-1).astype(jnp.float32)
        pred = jnp.argmax(logits, axis=-1)
        C_self = -jnp.mean(jnp.max(logp, axis=-1), axis=-1).astype(jnp.float32)
        yhat = jax.lax.stop_gradient(pred)
        grad_s = jax.grad(cost_batch)(s, yhat)
        G_self = jnp.linalg.norm(grad_s.reshape(B, -1), axis=1).astype(jnp.float32)
        res = S.free_residual(theta, x_inj, z).astype(jnp.float32)
        y_new = M.answer_update(y, s, y_rate)
        d = jnp.linalg.norm((y_new - y).reshape(B, -1), axis=1)
        dy = (d / (jnp.linalg.norm(y_new.reshape(B, -1), axis=1) + 1e-12)).astype(jnp.float32)
        return y_new, z, (probs, dy, res, C_self, G_self)

    y0 = jnp.zeros((B, M.P, C), theta["Ws"].dtype)
    z0 = M.init_state(theta, y0)
    y, z, out0 = one(y0, z0, True)
    out0 = tuple(o[None] for o in out0)
    if stages == 1:
        outs = out0
    else:
        def body(carry, _):
            y, z = carry
            y, z, out = one(y, z, False)
            return (y, z), out
        _, scanned = jax.lax.scan(body, (y, z), None, length=stages - 1)
        outs = tuple(jnp.concatenate([a, b]) for a, b in zip(out0, scanned))
    probs, dy, res, c_self, g_self = outs
    return dict(probs=probs, dy=dy, res=res, C_self=c_self, G_self=g_self)


def chain(params, puzzles, stages, cfg):
    """The free chain (no label, no nudge) for `stages` stages, the answer state
    carried exactly as training (cold first stage, then warm). puzzles: int
    (B, P) ints, 0 = blank, 1..N_CLASSES = clue. Returns a dict of per-stage
    stacked arrays: "probs" (stages, B, P, N_CLASSES) float32, "dy" (stages, B)
    the relative answer change, "res" (stages, B) the free state's one-sweep
    relative residual (solvers.free_residual), and two label-free test-time
    scores: "C_self" (stages, B), the readout cost of the network's own argmax,
    and "G_self" (stages, B), the L2 norm of that cost's gradient at the free
    equilibrium."""
    return _chain_jit(params, puzzles, stages=stages, m=cfg["m"], T_free=cfg["T_free"],
                      T_free_warm=cfg["T_free_warm"], y_rate=cfg["y_rate"])


def eval_chain(params, puzzles, solutions, stages, cfg, chunk=1000):
    """`chain`'s per-stage argmax accuracy against `solutions` (int, 1..
    N_CLASSES), over the whole set in chunks (sample-weighted means). Returns
    (stage_cell (stages,), stage_board (stages,)) numpy arrays."""
    labels = np.asarray(solutions) - 1
    n = puzzles.shape[0]
    cell_sum = np.zeros(stages, dtype=np.float64)
    board_sum = np.zeros(stages, dtype=np.float64)
    for i in range(0, n, chunk):
        out = chain(params, puzzles[i:i + chunk], stages, cfg)
        probs = np.asarray(out["probs"])                   # (stages, b, P, NC)
        y = labels[i:i + chunk]                             # (b, P)
        pred = probs.argmax(axis=-1)                         # (stages, b, P)
        correct = pred == y[None]
        b = y.shape[0]
        cell_sum += correct.mean(axis=(1, 2)) * b
        board_sum += correct.all(axis=2).mean(axis=1) * b
    return cell_sum / n, board_sum / n


# ------------------------------ checkpoints ------------------------------
def save_checkpoint(path, params, cfg, **meta):
    """{"params": ..., "cfg": cfg, **meta} pickled atomically (written to a temp
    file, then renamed)."""
    d = {"params": jax.tree_util.tree_map(np.asarray, params), "cfg": cfg, **meta}
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(d, f)
    os.replace(tmp, path)


def load_checkpoint(path):
    """-> (params, cfg) of a checkpoint written by `train` (best.pkl / last.pkl). The
    stored cfg must contain every configuration key (KeyError otherwise, rather than a
    silent fallback to the recipe); model.configure(cfg) is the caller's job."""
    with open(path, "rb") as f:
        ck = pickle.load(f)
    missing = sorted(set(DEFAULTS) - set(ck["cfg"]))
    if missing:
        raise KeyError(f"{path}: the checkpoint cfg lacks the configuration keys "
                       f"{missing}; it was not written by recursive.train")
    params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    return params, merge_cfg(ck["cfg"])


def _save_last(path, params, opt_state, ema, cfg, ep, hist):
    """last.pkl: the raw training params, the optimiser state as a flat
    leaf list (+ its treedef string, for reference), the raw EMA accumulator
    (present only when EMA is on), cfg, epoch and history."""
    d = {"params": jax.tree_util.tree_map(np.asarray, params), "cfg": cfg, "ep": ep,
        "history": hist,
        "opt_state_leaves": [np.asarray(a) for a in jax.tree_util.tree_leaves(opt_state)],
        "opt_state_treedef": str(jax.tree_util.tree_structure(opt_state))}
    if ema is not None:
        d["ema_params"] = jax.tree_util.tree_map(np.asarray, ema.params)
        d["ema_steps"] = ema.steps
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(d, f)
    os.replace(tmp, path)


# ------------------------------ training ------------------------------
def train(cfg, train_puzzles, train_solutions, valid_puzzles, valid_solutions, *,
          outdir, tag, augment_fn=None, log=print):
    """The deep-supervision training loop. train_puzzles/valid_puzzles: int
    (N, P), 0 = blank, 1..N_CLASSES = clue; train_solutions/valid_solutions: int
    (N, P) in 1..N_CLASSES (labels = solutions - 1 are computed per batch, after
    augmentation). augment_fn, if given, is applied to every training batch as
    augment_fn(key, puzzles, solutions) -> (puzzles, solutions) with jnp int
    (bs, P) arrays in and out, solutions still in 1..N_CLASSES. The first
    cfg["n_valid"] rows of the valid set (0 = all) are evaluated at
    cfg["eval_stages"] stages every cfg["report_every"] epochs (and after the
    first). Writes best.pkl (banked on the last stage's board accuracy, ties on
    cell), last.pkl and log.json under outdir; returns the path of best.pkl.
    `tag` only labels the log lines."""
    c = merge_cfg(cfg)
    set_matmul_precision(c)
    os.makedirs(outdir, exist_ok=True)
    log(f"[{tag}] {json.dumps(c)}")

    M.configure(c)

    PTR = jnp.asarray(train_puzzles)
    STR = jnp.asarray(train_solutions)
    n_valid = int(c["n_valid"])
    PVA = jnp.asarray(valid_puzzles)
    SVA = jnp.asarray(valid_solutions)
    if n_valid > 0:
        PVA, SVA = PVA[:n_valid], SVA[:n_valid]

    NTR = PTR.shape[0]
    BS = int(c["bs"])
    EP_N = min(NTR, int(c["ep_size"]) or NTR)

    params = M.init_params(jax.random.PRNGKey(int(c["seed"])), c)
    params = M.project_spectral(params, cap=c["cap"], n_iter=8)
    n_params = M.count_params(params)
    log(f"[{tag}] params total={n_params}")

    spe = EP_N // BS
    total_steps = int(round(c["epochs"] * spe * (c["n_sup_min"] + c["n_sup_max"]) / 2.0))
    opt = OPT.build_optimizer(c, total_steps)
    opt_state = opt.init(params)

    y0 = jnp.zeros((BS, M.P, int(c["C"])), params["theta"]["Ws"].dtype)
    z0 = M.init_state(params["theta"], y0)
    sched = parse_prune_sched(c["prune_sched"])
    rng_depth = np.random.default_rng(int(c["seed"]))
    key = jax.random.PRNGKey(int(c["seed"]) + 1)

    ema = OPT.EMA(c["ema"]) if c["ema"] > 0.0 else None
    proj = jax.jit(lambda p: M.project_spectral(p, cap=c["cap"], n_iter=8))
    best_cell, best_board = 0.0, 0.0
    bank_board, bank_cell = -1.0, -1.0
    hist = []
    t_run0 = time.time()
    best_path = os.path.join(outdir, "best.pkl")
    last_path = os.path.join(outdir, "last.pkl")

    for ep in range(int(c["epochs"])):
        key, kp = jax.random.split(key)
        perm = jax.random.permutation(kp, NTR)
        t0 = time.time()
        acc_logs, n_loops_ep = [], []
        for i in range(spe):
            idx = perm[i * BS:(i + 1) * BS]
            if augment_fn is not None:
                key, ka = jax.random.split(key)
                px, sx = augment_fn(ka, PTR[idx], STR[idx])
            else:
                px, sx = PTR[idx], STR[idx]
            py = sx - 1
            y_carry, z_warm = y0, z0
            n_loops = draw_n_loops(c, rng_depth)
            n_loops_ep.append(n_loops)
            for n in range(n_loops):
                key, ks = jax.random.split(key)
                prune_k = prune_rows_for_loop(sched, n, BS)
                params, opt_state, y_carry, z_warm, logs = loop_step(
                    params, opt, opt_state, px, py, y_carry, z_warm, ks,
                    cold=(n == 0), prune_k=prune_k, cfg=c)
                if ema is not None:
                    ema.update(params)
                acc_logs.append(logs)
        # one host transfer of the logs per epoch, not one per loop
        acc_logs = [{k: float(v) for k, v in d.items()} for d in acc_logs]
        t_train = time.time() - t0
        if not math.isfinite(acc_logs[-1]["loss"]):
            log(f"[{tag}] ep{ep+1} NON-FINITE loss -> stop")
            break
        re_ = c["report_every"] or 1
        if (ep + 1) % re_ == 0 or ep == 0:
            eval_params = params
            if ema is not None:
                eval_params = proj(ema.corrected())
            cells, boards = eval_chain(eval_params, PVA, SVA, int(c["eval_stages"]), c,
                                       chunk=2000)
            cell, board = float(cells[-1]), float(boards[-1])
            mean_key = lambda k: float(np.mean([d[k] for d in acc_logs]))
            row = dict(ep=ep + 1, cell=round(cell, 4), board=round(board, 4),
                      stage_cell=[round(float(v), 4) for v in cells],
                      stage_board=[round(float(v), 4) for v in boards],
                      e_free=round(mean_key("e_free"), 4), loss=round(mean_key("loss"), 4),
                      dy_train=round(mean_key("dy"), 4),
                      gnorm=round(mean_key("gnorm_theta"), 4),
                      prune_rows=round(mean_key("prune_rows"), 2),
                      n_loops=round(float(np.mean(n_loops_ep)), 3),
                      s_ep=round(t_train, 1), t_wall=round(time.time() - t_run0, 1))
            banked = (board > bank_board) or (board == bank_board and cell > bank_cell)
            if banked or not hist:
                bank_board, bank_cell = board, cell
                save_checkpoint(best_path, eval_params, c, ep=ep + 1, cell=cell, board=board,
                                stage_cell=row["stage_cell"], stage_board=row["stage_board"])
            best_cell, best_board = max(best_cell, cell), max(best_board, board)
            hist.append(row)
            log(f"[{tag}] {json.dumps(row)} best_board={best_board:.4f} "
                f"({time.time()-t0:.0f}s/ep)")
            try:
                _save_last(last_path, params, opt_state, ema, c, ep + 1, hist)
            except Exception as ex:          # noqa: BLE001 (never kill a run for it)
                log(f"[{tag}] WARNING: could not write last.pkl: {ex!r}")

    out = dict(tag=tag, cfg=c, best_cell=best_cell, best_board=best_board, hist=hist,
               n_params=n_params)
    with open(os.path.join(outdir, "log.json"), "w") as f:
        json.dump(out, f, indent=2)
    log(f"[{tag}] DONE best_cell={best_cell:.4f} best_board={best_board:.4f}")
    return best_path
