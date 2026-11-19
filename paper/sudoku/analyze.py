"""Table 6 (Sudoku, board accuracy in %) from the view files written by paper.sudoku.views:
one view, the lowest-self-cost view among K, and the oracle (any of the K views correct),
for the standard and the recursive model. Prints mean +- s.d. (ddof = 1) over seeds and
the per-seed values.

    python -m paper.sudoku.analyze table6 [--std_dir results/sudoku/standard_table6] \\
        [--rec_dir results/sudoku/recursive_table6] [--std_at 300] [--rec_at 32]

Self-cost selection: the view with the lowest C_self (non-finite = +inf), ties to the
first view.
"""
import argparse
import glob
import os
import re

import numpy as np


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def select_view(vcost, vc):
    """vcost (..., V) self-cost, vc (..., V) correct -> correct of the lowest-cost view
    (non-finite cost = +inf, ties to the first view)."""
    e = np.where(np.isfinite(vcost), vcost, np.inf)
    return np.take_along_axis(vc, e.argmin(-1)[..., None], -1)[..., 0]


def load_views(run_dir, at=None):
    """views/s<seed>_v<a>_<b>.npz of a run folder, per seed, with contiguous views from 0.
    Returns dict(seeds, idx, ck, V, view0/sel/oracle (n,nck,N) bool: first view, lowest-
    self-cost view, any view; vc (n,N,V) bool and vcost (n,N,V) f32: correctness and
    self-cost of every view at the checkpoint iteration `at` (default: the last))."""
    pat = re.compile(r"s(\d+)_v(\d+)_(\d+)\.npz")
    parts = {}
    for f in glob.glob(os.path.join(run_dir, "views", "s*_v*_*.npz")):
        m = pat.fullmatch(os.path.basename(f))
        if m:
            parts.setdefault(int(m.group(1)), []).append((int(m.group(2)), int(m.group(3)), f))
    if not parts:
        raise FileNotFoundError(f"no view files in {run_dir}/views")
    seeds = sorted(parts)
    out = dict(seeds=seeds, view0=[], sel=[], oracle=[], vc=[], vcost=[])
    idx = ck = None
    for s in seeds:
        found, pos, vcs, vcosts = sorted(parts[s]), 0, [], []
        for a, b, f in found:
            assert a == pos, f"seed {s}: gap or overlap at view {pos} (next file starts at {a})"
            pos = b
            z = np.load(f)
            if idx is None:
                idx, ck = z["idx"], [int(k) for k in z["ckpt_iters"]]
            assert np.array_equal(idx, z["idx"]) and ck == [int(k) for k in z["ckpt_iters"]], f
            vcs.append(z["view_correct"])
            vcosts.append(z["view_cost"])
        vc, vcost = np.concatenate(vcs, axis=-1), np.concatenate(vcosts, axis=-1)  # (nck,N,V)
        if "V" in out:
            assert out["V"] == pos, f"seed {s}: {pos} views, expected {out['V']}"
        out["V"] = pos
        out["view0"].append(vc[..., 0])
        out["sel"].append(select_view(vcost, vc))
        out["oracle"].append(vc.any(-1))
        k = len(ck) - 1 if at is None else ck.index(at)
        out["vc"].append(vc[k])
        out["vcost"].append(vcost[k])
    for key in ("view0", "sel", "oracle", "vc", "vcost"):
        out[key] = np.stack(out[key])
    out.update(idx=idx, ck=ck, at=ck[-1] if at is None else at)
    return out


# ---------------------------------------------------------------------------
# statistics and printing
# ---------------------------------------------------------------------------
def sd_over_seeds(v):
    """Sample standard deviation over the seed axis 0 (zeros for a single seed)."""
    return v.std(0, ddof=1) if v.shape[0] > 1 else np.zeros(v.shape[1:])


def _pm(x):
    """(n, c) percent values -> list of 'mean +- sd' strings."""
    mu, sd = x.mean(0), sd_over_seeds(x)
    return [f"{a:5.1f}+-{b:.1f}" for a, b in zip(mu, sd)]


def _row(label, cells, w=22):
    return f"{label:<{w}}" + " ".join(f"{c:>12}" for c in cells)


def table6(std_dir, rec_dir, std_at=300, rec_at=32):
    """Board accuracy (%) per seed with one view, self-cost selection and oracle, at one
    checkpoint iteration. Returns {row label: (n_seeds, 3) array}."""
    rows = {}
    for label, d, at in (("Standard", std_dir, std_at), ("Recursive", rec_dir, rec_at)):
        if not os.path.isdir(os.path.join(d, "views")):
            print(f"{d}: no views, skipping {label}")
            continue
        v = load_views(d, at=at)
        k = v["ck"].index(at)
        rows[f"{label} ({at} {'sweeps' if label == 'Standard' else 'stages'})"] = 100 * np.stack(
            [v["view0"][:, k].mean(1), v["sel"][:, k].mean(1), v["oracle"][:, k].mean(1)], 1)
        print(f"{label}: seeds {v['seeds']}, {len(v['idx'])} boards, {v['V']} views")
    return rows


def print_table6(rows):
    """Print the rows of `table6`: mean +- s.d. over seeds, then the per-seed values."""
    print(_row("", ["one view", "self-cost", "oracle"], 28))
    for label, x in rows.items():
        print(_row(label, _pm(x), 28))
        print(f"   per seed (one view / self-cost / oracle): "
              + " | ".join("/".join(f"{c:.2f}" for c in r) for r in x))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("table6", help="Table 6: one view / self-cost / oracle board accuracy")
    b.add_argument("--std_dir", default="results/sudoku/standard_table6",
                   help="output folder of paper.sudoku.views for the standard model")
    b.add_argument("--rec_dir", default="results/sudoku/recursive_table6",
                   help="output folder of paper.sudoku.views for the recursive model")
    b.add_argument("--std_at", type=int, default=300,
                   help="sweep of the standard model (one of its stored iterations)")
    b.add_argument("--rec_at", type=int, default=32,
                   help="stage of the recursive model (one of its stored iterations)")
    args = ap.parse_args()
    print_table6(table6(args.std_dir, args.rec_dir, args.std_at, args.rec_at))


if __name__ == "__main__":
    main()
