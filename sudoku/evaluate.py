"""Evaluation of the standard Sudoku model: the two readouts (canonical and damped
"equilibrium") used for checkpoint selection during training and for the test-set report,
and K-view digit-relabelling test-time augmentation (K = 64 in Table 6) with three
selectors (validity / self-cost / oracle).

Two readouts, each read off the main value node s:
  canonical   = plain free phase (core.ep.free_phase, solver="sweep"), T=150 sweeps
                from z = 0.
  equilibrium = damped fixed-point readout: s <- (1-a) s + a block_sweep(z)[-1], every
                other value node taking the fresh block_sweep value at each iteration,
                T=300 sweeps, a=0.5, from z = 0. With K = 1 this is the single-view
                damped readout (x4 <- 0.5 x4 + 0.5 x4_hat).

    python -m sudoku.evaluate --ckpt runs/sudoku/s0/best_eq.pkl [--views 64]
                              [--split test|valid] [--readout auto|canonical|equilibrium|both]
                              [--sweeps 300] [--damping 0.5] [--n N] [--data_dir DIR]

prints, per readout, the single-pass board / cell accuracy and, over --views views, the
board accuracy of the validity selector (first view whose prediction is a valid grid
reproducing the clues), of the self-cost selector (lowest C_self) and of the oracle (any
view correct). --views 1 gives the single-view number.
"""
import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from core import ep as E
from . import data as Dat
from . import model as M

T_CANON = 150
EQ_A, EQ_T = 0.5, 300
K_VIEWS = 64
VIEW_SEED = 1000


# ------------------------------------------------------------------ checkpoints
def load_checkpoint(path):
    """Load a checkpoint written by sudoku.train (best_canon.pkl / best_eq.pkl /
    final.pkl) and configure sudoku.model from its cfg. Returns (params, cfg, readouts):
    readouts = the readout the checkpoint was selected on ("canonical" for
    best_canon.pkl, "equilibrium" for best_eq.pkl), or both for final.pkl."""
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    cfg = blob.get("cfg")
    if not cfg:                       # a checkpoint without its cfg: the training recipe
        from sudoku.train import RECIPE
        cfg = dict(RECIPE)
    M.configure(cfg)
    params = jax.tree_util.tree_map(jnp.asarray, blob["params"])
    which = {"canon": "canonical", "eq": "equilibrium"}.get(blob.get("which"), blob.get("which"))
    readouts = (which,) if which in ("canonical", "equilibrium") else ("canonical", "equilibrium")
    return params, cfg, readouts


# ------------------------------------------------------------------ readouts
def canonical_forward(theta, x_inj):
    """z* = plain free phase, T=150, from z=0 (core.ep.free_phase, solver="sweep")."""
    return E.free_phase(M, theta, x_inj, {"solver": "sweep", "T": T_CANON})


def canon_eval(params, X, Y):
    """(cell, board) on (X, Y) under the plain T=150 sweep readout."""
    cell, board = E.ep_eval(M, params, X, Y, {"solver": "sweep", "T": T_CANON})
    return float(cell), float(board)


def damped_forward(theta, x_inj, a=EQ_A, T=EQ_T):
    """z* = T sweeps of s <- (1-a)*s + a*block_sweep(z)[-1] from z=0; every
    other value node takes the fresh block_sweep value each iteration (only s is
    damped)."""
    def body(_, z):
        zs = M.block_sweep(theta, z, x_inj)
        return zs[:-1] + ((1.0 - a) * z[-1] + a * zs[-1],)
    return jax.lax.fori_loop(0, T, body, M.init_state(theta, x_inj))


def damped_eval(params, X, Y, a=EQ_A, T=EQ_T):
    """(cell, board) on (X, Y) under the damped a/T readout."""
    theta, psi = params["theta"], params["psi"]
    z = damped_forward(theta, M.embed(theta, X), a, T)
    cell, board = M.accuracy(psi, z, Y)
    return float(cell), float(board)


FORWARD = {"canonical": canonical_forward, "equilibrium": damped_forward}
READOUT_KW = {"canonical": {}, "equilibrium": dict(a=EQ_A, T=EQ_T)}


def readout_logits(which, params, X):
    """Per-cell logits (B, P, N_CLASSES) of readout `which` ("canonical" |
    "equilibrium") on the un-permuted puzzle X."""
    theta, psi = params["theta"], params["psi"]
    z = FORWARD[which](theta, M.embed(theta, X), **READOUT_KW[which])
    return M.readout(psi, z[-1])


def single_pass(which, params, puz, sol, chunk=2000):
    """Single pass of readout `which` over the whole split, in chunks: exact board and
    cell accuracy over all puzzles (counts summed over chunks)."""
    n = puz.shape[0]
    nb = nc = 0
    for j in range(0, n, chunk):
        X = jnp.asarray(puz[j:j + chunk])
        Y = jnp.asarray(sol[j:j + chunk] - 1)
        k = int(X.shape[0])
        theta, psi = params["theta"], params["psi"]
        z = FORWARD[which](theta, M.embed(theta, X), **READOUT_KW[which])
        c_, b_ = map(float, M.accuracy(psi, z, Y))
        nb += int(round(b_ * k)); nc += int(round(c_ * k * z[-1].shape[1]))
    P = puz.shape[1]
    return dict(board=nb / n, cell=nc / (P * n), n_board=nb, n_cell=nc, n=n)


# ------------------------------------------------------------------ TTA / verifiers
def grid_valid_np(pred, puz):
    """numpy: pred (n,81) classes 0..8, puz (n,81) 0..9 -> (n,) bool: every
    row/col/box holds each digit once AND every clue is reproduced."""
    n = pred.shape[0]
    oh = np.eye(9, dtype=np.int32)[pred].reshape(n, 9, 9, 9)
    rows = np.all(oh.sum(axis=2) == 1, axis=(1, 2))
    cols = np.all(oh.sum(axis=1) == 1, axis=(1, 2))
    boxes = np.all(oh.reshape(n, 3, 3, 3, 3, 9).sum(axis=(2, 4)) == 1, axis=(1, 2, 3))
    clues = np.all((puz == 0) | (pred == puz - 1), axis=1)
    return rows & cols & boxes & clues


def view_perms(K=K_VIEWS, seed=VIEW_SEED):
    """(K, 9) int64: row 0 = identity, rows 1..K-1 = distinct non-identity
    permutations of the classes 0..8, drawn once with
    np.random.RandomState(seed) -- the SAME K views for every puzzle/model."""
    rng = np.random.RandomState(seed)
    ident = tuple(range(9))
    out, seen = [ident], {ident}
    while len(out) < K:
        p = tuple(int(x) for x in rng.permutation(9))
        if p not in seen:
            seen.add(p)
            out.append(p)
    return np.asarray(out, np.int64)


def apply_view(grid, perm):
    """digit grid (0 = blank, 1..9) -> relabeled grid: d -> perm[d-1] + 1, 0
    stays 0 (same dtype)."""
    dmap = np.concatenate([[0], perm + 1]).astype(grid.dtype)
    return dmap[grid]


def _cself(logits):
    """C_self = mean over cells of -log max_c softmax(readout)_c (the model's
    own confidence in its answer, independent of ground truth)."""
    return -jnp.mean(jnp.max(jax.nn.log_softmax(logits, axis=-1), axis=-1), axis=1)


def make_forward(which, params):
    """x (B,81) -> (pred (B,81) argmax classes in x's own frame, C_self (B,))."""
    cself_j = jax.jit(_cself)

    def fwd(x):
        logits = readout_logits(which, params, x)
        return jnp.argmax(logits, axis=-1), cself_j(logits)
    return jax.jit(fwd)


def select(rv, Kk):
    """Verifier selections over the first Kk views of rv (corr/valid/cellc/cself,
    shape (n, K)): validity verifier, self-cost verifier, oracle."""
    c = rv["corr"][:, :Kk]
    v = rv["valid"][:, :Kk]
    cc = rv["cellc"][:, :Kk].astype(np.int64)
    s = rv["cself"][:, :Kk].astype(np.float64)
    s = np.where(np.isfinite(s), s, np.inf)
    n = c.shape[0]
    ii = np.arange(n)
    anyv = v.any(axis=1)
    kv = np.where(anyv, v.argmax(axis=1), 0)             # first valid view, else view 0
    kc = s.argmin(axis=1)                                 # lowest C_self, ties -> earliest view
    anyc = c.any(axis=1)
    return dict(K=int(Kk),
                valid_board=float(c[ii, kv].mean()), valid_cell=float(cc[ii, kv].sum() / (81.0 * n)),
                cost_board=float(c[ii, kc].mean()), cost_cell=float(cc[ii, kc].sum() / (81.0 * n)),
                oracle_board=float(anyc.mean()),
                answered_valid=float(anyv.mean()),
                valid_but_wrong_pairs=int((v & ~c).sum()),
                mean_first_valid_view=(float(kv[anyv].mean()) if anyv.any() else None),
                cost_pick_not_view0=float((kc != 0).mean()))


def auc_lower(score, label):
    """P(score of a correct item < score of a wrong item), ties count 1/2
    (lower cost = correct) -- the cost verifier's discrimination (AUC)."""
    score = np.asarray(score, np.float64)
    label = np.asarray(label, bool)
    pos, neg = np.sort(score[label]), np.sort(score[~label])
    if len(pos) == 0 or len(neg) == 0:
        return None
    lo = np.searchsorted(neg, pos, "left")
    hi = np.searchsorted(neg, pos, "right")
    return float(((len(neg) - hi) + 0.5 * (hi - lo)).sum() / (float(len(pos)) * len(neg)))


def run_views(which, params, puz, sol, perms, chunk=2000):
    """Run all K views of `perms` through readout `which`; returns
    corr/valid/cellc/cself arrays of shape (n, K)."""
    fwd = make_forward(which, params)
    n, Kv = puz.shape[0], perms.shape[0]
    y = (sol - 1).astype(np.int64)
    corr = np.zeros((n, Kv), bool)
    valid = np.zeros((n, Kv), bool)
    cellc = np.zeros((n, Kv), np.int16)
    cself = np.zeros((n, Kv), np.float32)
    for k in range(Kv):
        perm = perms[k]
        inv = np.argsort(perm)
        pv = apply_view(puz, perm)
        for j in range(0, n, chunk):
            pred, cs = fwd(jnp.asarray(pv[j:j + chunk]))
            po = inv[np.asarray(pred)]                      # back to the original digit frame
            hit = po == y[j:j + chunk]
            corr[j:j + chunk, k] = hit.all(axis=1)
            cellc[j:j + chunk, k] = hit.sum(axis=1)
            valid[j:j + chunk, k] = grid_valid_np(po, puz[j:j + chunk])
            cself[j:j + chunk, k] = np.asarray(cs)
    return dict(corr=corr, valid=valid, cellc=cellc, cself=cself)


def evaluate_readout(which, params, puz, sol, K=K_VIEWS, chunk=2000):
    """Full report for one readout on (puz, sol): single pass, view-0-vs-single
    sanity, K-view verifier scores (validity / cost / oracle), AUC of the cost
    verifier. puz/sol: numpy (n, 81), puzzle 0=blank/1..9, solution 1..9."""
    perms = view_perms(K)
    sp = single_pass(which, params, puz, sol, chunk)
    rv = run_views(which, params, puz, sol, perms, chunk)
    k_all = select(rv, K)
    cs = np.where(np.isfinite(rv["cself"]), rv["cself"], np.inf).astype(np.float64)
    return dict(single=sp, k=k_all, view0_board=float(rv["corr"][:, 0].mean()),
                auc_cost_view0=auc_lower(cs[:, 0], rv["corr"][:, 0]),
                auc_cost_pooled=auc_lower(cs.ravel(), rv["corr"].ravel()))


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description="Evaluate a standard Sudoku checkpoint: "
                                             "single pass and K-view test-time selection.")
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint written by sudoku.train (best_canon.pkl, best_eq.pkl "
                         "or final.pkl)")
    ap.add_argument("--views", type=int, default=K_VIEWS,
                    help="number K of digit-relabelled views (1 = single view; Table 6: 64)")
    ap.add_argument("--split", default="test", choices=("test", "valid"),
                    help="split to evaluate on")
    ap.add_argument("--readout", default="auto",
                    choices=("auto", "canonical", "equilibrium", "both"),
                    help="auto = the readout the checkpoint was selected on (both for "
                         "final.pkl); equilibrium = the damped readout "
                         "s <- (1 - damping) s + damping s_hat")
    ap.add_argument("--sweeps", type=int, default=EQ_T,
                    help="sweeps of the damped readout (default 300, as in training)")
    ap.add_argument("--damping", type=float, default=EQ_A,
                    help="damping a of the damped readout (default 0.5)")
    ap.add_argument("--n", type=int, default=0,
                    help="evaluate the first n boards of the split only (0 = all)")
    ap.add_argument("--data_dir", default=Dat.RRN_DIR,
                    help="directory with <split>.npz written by sudoku.prepare "
                         "(default: data/sudoku in the repository, or $SUDOKU_RRN_DIR)")
    args = ap.parse_args()
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")

    params, cfg, readouts = load_checkpoint(args.ckpt)
    if args.readout != "auto":
        readouts = (("canonical", "equilibrium") if args.readout == "both"
                    else (args.readout,))
    READOUT_KW["equilibrium"] = dict(a=args.damping, T=args.sweeps)
    puz, sol = Dat.load_split(args.split, args.data_dir)
    if args.n:
        puz, sol = puz[:args.n], sol[:args.n]
    print(f"evaluating {args.ckpt} on {args.split} split ({puz.shape[0]} puzzles), "
          f"K={args.views}, readouts={readouts}", flush=True)
    for ro in readouts:
        r = evaluate_readout(ro, params, puz, sol, K=args.views)
        print(f"[{ro}] single board={r['single']['board']:.4f} cell={r['single']['cell']:.4f} | "
              f"K={args.views} validity={r['k']['valid_board']:.4f} "
              f"cost={r['k']['cost_board']:.4f} oracle={r['k']['oracle_board']:.4f} | "
              f"auc_cost_view0={r['auc_cost_view0']}", flush=True)
        assert abs(r["view0_board"] - r["single"]["board"]) < 1e-9, (
            f"view 0 (identity) board accuracy {r['view0_board']} differs from the single "
            f"pass {r['single']['board']}")


if __name__ == "__main__":
    main()
