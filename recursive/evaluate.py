"""Evaluation of a recursive circular model checkpoint (best.pkl written by
recursive.train) on the Sudoku RRN validation / test split.

The free chain runs --stages stages (default 32) with the free-phase budget stored in
the checkpoint (T_free = 600 iterations at the first stage, T_free_warm = 80 at later
stages); --t_free / --t_warm override it.

--views 1 (default): fixed depth. Run the free chain once and report, for every stage k
in --report_at, the board / cell accuracy of the stage-k prediction and the fraction of
puzzles whose stage-k free phase has converged (one-sweep relative residual < 1e-3).

--views K > 1: label-free digit-relabelling test-time augmentation. View 0 is the puzzle
itself; view k >= 1 relabels every clue digit by a fixed permutation of 1..9 (drawn once
per view, recursive.data.build_perms). The model runs its own free chain on every view
and the prediction is mapped back to the original digits before anything is scored. Per
view and stage this keeps C_self (the readout cost of the network's own answer), G_self
(the norm of the gradient of that cost at the readout state), the free-phase residual
and the label-free validity check of the predicted grid. These feed label-free board
selectors over K in {1, 2, 4, 8, 16, 32, 64} views: verifier_* (first valid grid),
cost_* (lowest C_self), gradC_* (lowest G_self), vote_last (cellwise plurality), and the
label-reading oracle_* (any view correct, a ceiling only). *_last selectors read the last
stage, *_any ones every (view, stage >= --min_stage) pair.

Both modes run at the checkpoint's matmul_precision, set before any computation.

    python -m recursive.evaluate --ckpt runs/recursive/s0/best.pkl --split test
    python -m recursive.evaluate --ckpt runs/recursive/s0/best.pkl --split test --views 64 \\
        [--out_dir DIR] [--summary FILE]
"""
import argparse
import json
import os

import numpy as np

from recursive import data, model, pipeline
from sudoku import data as sudoku_data
from sudoku.evaluate import grid_valid_np

K_GRID = (1, 2, 4, 8, 16, 32, 64)
RES_TOL = 1e-3


def _fmt(v):
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _print_table(rows, cols):
    print("  ".join(f"{c:>12}" for c in cols))
    for r in rows:
        print("  ".join(f"{_fmt(r.get(c)):>12}" for c in cols))


# ============================================================================
# fixed-depth protocol
# ============================================================================
def fixed_depth(params, cfg, puzzles, solutions, stages, report_at, chunk):
    """One pass of the free chain (recursive.pipeline.chain, `chunk` puzzles at a time)
    for `stages` stages. For every stage k in `report_at` (1-based, <= stages):
    board / cell accuracy of the stage-k prediction (argmax of the readout)
    against solutions - 1, and frac_conv, the fraction of puzzles whose stage-k
    free state has one-sweep relative residual < RES_TOL. solutions: (n, 81) in
    1..9. Returns a list of dicts {k, board, cell, frac_conv}."""
    labels = np.asarray(solutions) - 1
    n = puzzles.shape[0]
    cell_sum = np.zeros(stages, dtype=np.float64)
    board_sum = np.zeros(stages, dtype=np.float64)
    conv_sum = np.zeros(stages, dtype=np.float64)
    for i in range(0, n, chunk):
        out = pipeline.chain(params, puzzles[i:i + chunk], stages, cfg)
        pred = np.asarray(out["probs"]).argmax(axis=-1)            # (stages, b, 81)
        res = np.asarray(out["res"])                                # (stages, b)
        correct = pred == labels[i:i + chunk][None]                 # (stages, b, 81)
        cell_sum += correct.mean(axis=2).sum(axis=1)
        board_sum += correct.all(axis=2).sum(axis=1)
        conv_sum += (res < RES_TOL).sum(axis=1)
    cell, board, frac_conv = cell_sum / n, board_sum / n, conv_sum / n
    return [dict(k=int(k), board=float(board[k - 1]), cell=float(cell[k - 1]),
                 frac_conv=float(frac_conv[k - 1])) for k in report_at]


# ============================================================================
# TTA protocol
# ============================================================================
def _run_view(params, cfg, puzzles, solutions, perm, stages, chunk):
    """One view over the whole puzzle set: relabel by `perm`, run the free chain
    (recursive.pipeline.chain, chunked here), map the prediction back to the original
    digit frame before anything is scored (original frame throughout). Returns
    dict(pred_last (n,81) int8, correct/valid (n,stages) bool, C/G/res (n,stages)
    float32)."""
    n = puzzles.shape[0]
    labels = solutions - 1
    pred_last = np.zeros((n, 81), dtype=np.int8)
    correct = np.zeros((n, stages), dtype=bool)
    valid = np.zeros((n, stages), dtype=bool)
    C = np.zeros((n, stages), dtype=np.float32)
    G = np.zeros((n, stages), dtype=np.float32)
    res = np.zeros((n, stages), dtype=np.float32)
    for i in range(0, n, chunk):
        puz = puzzles[i:i + chunk]
        lab = labels[i:i + chunk]
        puz_v = data.relabel(puz, perm)
        out = pipeline.chain(params, puz_v, stages, cfg)
        probs = np.asarray(out["probs"])                           # (stages,b,81,9)
        pred_v = probs.argmax(axis=-1)                              # permuted frame
        pred_o = data.unrelabel_pred(pred_v, perm)                  # original frame
        b = pred_o.shape[1]
        sl = slice(i, i + b)
        puz_rep = np.repeat(puz[None], stages, axis=0).reshape(stages * b, 81)
        vd = grid_valid_np(pred_o.reshape(stages * b, 81), puz_rep).reshape(stages, b)
        ok = np.all(pred_o == lab[None], axis=-1)                   # (stages, b)
        pred_last[sl] = pred_o[-1]
        correct[sl] = ok.T
        valid[sl] = vd.T
        C[sl] = np.asarray(out["C_self"]).T
        G[sl] = np.asarray(out["G_self"]).T
        res[sl] = np.asarray(out["res"]).T
    return dict(pred_last=pred_last, correct=correct, valid=valid, C=C, G=G, res=res)


def run_tta(params, cfg, puzzles, solutions, stages, views, view_seed, chunk, out_dir):
    """Runs every view in order, writing out_dir/view_{k:02d}.npz (idx, pred_last,
    correct, valid, C, G, res) and out_dir/labels.npz; prints one JSON line per
    view with the running board/valid rate of that view alone."""
    os.makedirs(out_dir, exist_ok=True)
    idx = np.arange(puzzles.shape[0], dtype=np.int32)
    np.savez(os.path.join(out_dir, "labels.npz"), lab=(solutions - 1))
    perms = data.build_perms(views, view_seed)
    for k in range(views):
        d = _run_view(params, cfg, puzzles, solutions, perms[k], stages, chunk)
        np.savez(os.path.join(out_dir, f"view_{k:02d}.npz"), idx=idx,
                 pred_last=d["pred_last"], correct=d["correct"].astype(np.uint8),
                 valid=d["valid"].astype(np.uint8), C=d["C"], G=d["G"],
                 res=d["res"].astype(np.float16))
        print(json.dumps(dict(view=int(k), perm=perms[k].tolist(), n=int(idx.shape[0]),
                              board_last=round(float(d["correct"][:, -1].mean()), 4),
                              valid_last=round(float(d["valid"][:, -1].mean()), 4))),
              flush=True)


def _vote(pred):
    """pred (K, n, 81) 0..8 -> (n, 81) cellwise plurality; ties -> the lowest view
    index whose own prediction is a max-count candidate."""
    k, n, P = pred.shape
    eye = np.eye(9, dtype=np.int32)
    counts = np.zeros((n, P, 9), dtype=np.int32)
    for s in range(k):
        counts += eye[pred[s]]
    cand = counts == counts.max(axis=-1, keepdims=True)
    ii, jj = np.arange(n)[:, None], np.arange(P)[None, :]
    vote, done = pred[0].copy(), np.zeros((n, P), dtype=bool)
    for s in range(k):
        take = cand[ii, jj, pred[s]] & ~done
        vote = np.where(take, pred[s], vote)
        done |= take
    return vote


def _load_views(out_dir):
    """The view_*.npz files written by run_tta, in view order."""
    views, k = [], 0
    while True:
        path = os.path.join(out_dir, f"view_{k:02d}.npz")
        if not os.path.exists(path):
            break
        views.append(dict(np.load(path)))
        k += 1
    return views


def analyze(out_dir, min_stage, clues=None):
    """Board accuracy of several label-free selectors (plus two label-reading
    oracle ceilings) over K in {1,2,4,...} <= the number of saved views, from
    out_dir/view_*.npz + labels.npz. Selectors: view0_last, verifier_last,
    cost_last, gradC_last, vote_last, oracle_last (+ *_any variants scanning
    (view, stage >= min_stage) pairs); the valid_but_wrong diagnostic; and a
    clue-bucket table at the largest K for view0_last / verifier_last /
    gradC_last / oracle_last (needs `clues`, an (n,) array of clue counts aligned
    to the puzzle index -- data.clue_counts). Returns a flat list of dicts: one
    row per K, one {"row": "diagnostics", ...}, and (clues given) one
    {"row": "clue_bucket", ...} per distinct clue count."""
    views = _load_views(out_dir)
    if not views:
        raise FileNotFoundError(f"no view_*.npz under {out_dir}")
    n = int(views[0]["idx"].shape[0])
    stages = int(views[0]["correct"].shape[1])
    V = len(views)
    lab = np.load(os.path.join(out_dir, "labels.npz"))["lab"]

    correct = np.stack([v["correct"] for v in views], axis=0).astype(bool)   # (V,n,stages)
    valid = np.stack([v["valid"] for v in views], axis=0).astype(bool)
    C = np.stack([v["C"] for v in views], axis=0).astype(np.float32)
    G = np.stack([v["G"] for v in views], axis=0).astype(np.float32)
    pred_last = np.stack([v["pred_last"] for v in views], axis=0)            # (V,n,81)

    window = slice(min_stage - 1, stages)
    ar = np.arange(n)
    cand_Ks = [k for k in K_GRID if k <= V]
    rows, bucket_vecs = [], None
    for K in cand_Ks:
        cl, vl = correct[:K, :, -1], valid[:K, :, -1]
        Cl, Gl = C[:K, :, -1], G[:K, :, -1]
        v0 = cl[0]
        has_v = vl.any(axis=0)
        first_v = vl.argmax(axis=0)                                # 0 if none valid
        verifier_last = cl[np.where(has_v, first_v, 0), ar]
        cost_last = cl[Cl.argmin(axis=0), ar]
        gradC_last = cl[Gl.argmin(axis=0), ar]
        oracle_last = cl.any(axis=0)
        vote_last = np.all(_vote(pred_last[:K]) == lab, axis=1)

        cw, vw = correct[:K, :, window], valid[:K, :, window]
        Cw, Gw = C[:K, :, window], G[:K, :, window]
        W = Cw.shape[-1]
        flat = lambda a: a.transpose(1, 0, 2).reshape(n, K * W)
        fc, fC, fG, fv = flat(cw), flat(Cw), flat(Gw), flat(vw)
        cost_any = fc[ar, fC.argmin(axis=1)]
        gradC_any = fc[ar, fG.argmin(axis=1)]
        has_va = fv.any(axis=1)
        verifier_any = np.where(has_va, fc[ar, fv.argmax(axis=1)], v0)
        oracle_any = cw.any(axis=(0, 2))

        rows.append(dict(K=int(K), n=int(n), view0_last=float(v0.mean()),
                         verifier_last=float(verifier_last.mean()),
                         cost_last=float(cost_last.mean()),
                         gradC_last=float(gradC_last.mean()),
                         vote_last=float(vote_last.mean()),
                         oracle_last=float(oracle_last.mean()),
                         verifier_any=float(verifier_any.mean()),
                         cost_any=float(cost_any.mean()),
                         gradC_any=float(gradC_any.mean()),
                         oracle_any=float(oracle_any.mean())))
        if K == cand_Ks[-1]:
            bucket_vecs = dict(view0_last=v0, verifier_last=verifier_last,
                               gradC_last=gradC_last, oracle_last=oracle_last)

    diag = dict(row="diagnostics",
               valid_but_wrong_last=int(np.sum(valid[:, :, -1] & ~correct[:, :, -1])),
               valid_count_last=int(np.sum(valid[:, :, -1])))

    bucket_rows = []
    if bucket_vecs is not None and clues is not None:
        clues_arr = np.asarray(clues)
        K_top = cand_Ks[-1]
        for cval in sorted(set(int(v) for v in clues_arr.tolist())):
            m = clues_arr == cval
            if not m.any():
                continue
            bucket_rows.append(dict(
                row="clue_bucket", K=K_top, clues=int(cval), n=int(m.sum()),
                view0_last=float(bucket_vecs["view0_last"][m].mean()),
                verifier_last=float(bucket_vecs["verifier_last"][m].mean()),
                gradC_last=float(bucket_vecs["gradC_last"][m].mean()),
                oracle_last=float(bucket_vecs["oracle_last"][m].mean())))
    return rows + [diag] + bucket_rows


def write_summary(rows, path, meta=None):
    """Plain-text (fixed-width, no Markdown) summary table written to `path`;
    returns the text."""
    sel_rows = [r for r in rows if "row" not in r]
    diag = next((r for r in rows if r.get("row") == "diagnostics"), {})
    buckets = [r for r in rows if r.get("row") == "clue_bucket"]
    lines = []
    if meta:
        lines.append(" ".join(f"{k}={v}" for k, v in meta.items()))
        lines.append("")
    lines.append("selector board accuracy (fraction of all n puzzles)")
    cols = ("K", "view0_last", "verifier_last", "cost_last", "gradC_last",
            "vote_last", "oracle_last", "verifier_any", "cost_any", "gradC_any",
            "oracle_any")
    lines.append("  ".join(f"{c:>14}" for c in cols))
    for r in sel_rows:
        lines.append("  ".join(f"{_fmt(r.get(c)):>14}" for c in cols))
    lines.append("")
    lines.append("diagnostics (pooled over every saved view)")
    for k, v in diag.items():
        if k == "row":
            continue
        lines.append(f"  {k:<22}{_fmt(v)}")
    if buckets:
        lines.append("")
        lines.append(f"clue-count breakdown (K={buckets[0]['K']})")
        bc = ("clues", "n", "view0_last", "verifier_last", "gradC_last", "oracle_last")
        lines.append("  ".join(f"{c:>14}" for c in bc))
        for r in sorted(buckets, key=lambda r: r["clues"]):
            lines.append("  ".join(f"{_fmt(r.get(c)):>14}" for c in bc))
    text = "\n".join(lines) + "\n"
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    open(path, "w").write(text)
    return text


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Evaluate a recursive circular model "
                                             "checkpoint on Sudoku (fixed depth or K views).")
    ap.add_argument("--ckpt", required=True, help="best.pkl written by recursive.train")
    ap.add_argument("--split", default="test", choices=["valid", "test"],
                    help="split to evaluate on")
    ap.add_argument("--data_dir", default=sudoku_data.RRN_DIR,
                    help="directory with <split>.npz written by sudoku.prepare (default: "
                         "data/sudoku in the repository, or $SUDOKU_RRN_DIR)")
    ap.add_argument("--n", type=int, default=0,
                    help="evaluate the first n boards of the split only (0 = all)")
    ap.add_argument("--stages", type=int, default=32, help="stages of the free chain")
    ap.add_argument("--t_free", type=int, default=None,
                    help="free-phase iterations of the first stage (default: the "
                         "checkpoint's T_free)")
    ap.add_argument("--t_warm", type=int, default=None,
                    help="free-phase iterations of the later stages (default: the "
                         "checkpoint's T_free_warm)")
    ap.add_argument("--report_at", default="1,4,8,12,16,24,32",
                    help="fixed depth: comma list of stages k to report (filtered to "
                         "k <= stages; --stages itself always included)")
    ap.add_argument("--views", type=int, default=1,
                    help="1 = fixed depth; > 1 = label-free test-time selection over that "
                         "many digit-relabelled views")
    ap.add_argument("--view_seed", type=int, default=1000,
                    help="seed of the view permutations (view k uses seed + k)")
    ap.add_argument("--min_stage", type=int, default=4,
                    help="views: first stage (1-based) admitted by the *_any selectors")
    ap.add_argument("--chunk", type=int, default=1000, help="boards per jitted call")
    ap.add_argument("--out_dir", default="",
                    help="views: directory for view_*.npz / labels.npz / summary.txt "
                         "(default: <ckpt without .pkl>_tta/)")
    ap.add_argument("--summary", default="",
                    help="views: summary table path (default: <out_dir>/summary.txt)")
    args = ap.parse_args()

    puzzles, solutions = sudoku_data.load_split(args.split, args.data_dir)
    if args.n:
        puzzles, solutions = puzzles[:args.n], solutions[:args.n]

    params, cfg = pipeline.load_checkpoint(args.ckpt)
    if args.t_free is not None:
        cfg["T_free"] = int(args.t_free)
    if args.t_warm is not None:
        cfg["T_free_warm"] = int(args.t_warm)
    print(f"free phase: T_free={cfg['T_free']} T_free_warm={cfg['T_free_warm']} "
          f"stages={args.stages}", flush=True)
    pipeline.set_matmul_precision(cfg)          # before any chain computation
    model.configure(cfg)

    ks = sorted({int(k) for k in args.report_at.split(",") if k.strip()
                and 1 <= int(k) <= args.stages} | {args.stages})

    if args.views <= 1:
        rows = fixed_depth(params, cfg, puzzles, solutions, args.stages, ks, args.chunk)
        _print_table(rows, ("k", "board", "cell", "frac_conv"))
        for r in rows:
            print(json.dumps(r), flush=True)
        return

    out_dir = args.out_dir or (os.path.splitext(args.ckpt)[0] + "_tta")
    run_tta(params, cfg, puzzles, solutions, args.stages, args.views, args.view_seed,
            args.chunk, out_dir)
    rows = analyze(out_dir, args.min_stage, clues=data.clue_counts(puzzles))
    summary_path = args.summary or os.path.join(out_dir, "summary.txt")
    meta = dict(ckpt=args.ckpt, split=args.split, n=int(puzzles.shape[0]),
               stages=args.stages, T_free=cfg["T_free"], T_free_warm=cfg["T_free_warm"],
               views=args.views, view_seed=args.view_seed, min_stage=args.min_stage)
    text = write_summary(rows, summary_path, meta=meta)
    print(text)
    for r in rows:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
