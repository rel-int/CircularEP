"""Train the recursive circular model on the Sudoku benchmark of Palm et al. 2018
(sudoku-hard RRN split: 180k train / 18k valid / 18k test boards, 17-34 givens).

The recursive ring is trained with deep supervision: every training batch (128 puzzles,
each under a fresh random Sudoku symmetry) runs a random number of loops drawn uniformly
in 4..16; each loop is one free relaxation, one fused +/-beta nudged pair, one centred EP
contrast and one optimiser step, with the answer state carried from loop to loop. From
the fifth loop of a batch on, the nudged phase and the contrast run on 64 of the 128
rows, chosen from the previous loop's answer (unsolved puzzles first, a quarter of the
slots kept for solved ones; prune_sched / prune_keep). 90 epochs of 8000 puzzles, EMA
0.999; the checkpoint is banked on the board accuracy of the first 2000 validation
puzzles at 16 stages. The recipe is CONFIG (= recursive.pipeline.DEFAULTS without the
grid size N and the seed).

    python -m recursive.train [--seed S] [--data_dir DIR] [--out DIR] [--set KEY=VALUE ...]

reads train.npz / valid.npz written by `python -m sudoku.prepare` (default data/sudoku in
the repository, or $SUDOKU_RRN_DIR) and writes best.pkl (the checkpoint read by
`python -m recursive.evaluate`), last.pkl and log.json to --out (default
runs/recursive/s<seed>). --set overrides CONFIG keys for short runs, e.g.
    python -m recursive.train --set epochs=1 ep_size=16 bs=8
"""
import argparse
import functools
import json
import os

from recursive import model, pipeline
from sudoku import data as sudoku_data

# The recipe: every configuration key except the grid size N and the seed (--seed).
CONFIG = {k: v for k, v in pipeline.DEFAULTS.items() if k not in ("N", "seed")}


def main():
    ap = argparse.ArgumentParser(description="Train the recursive circular model on "
                                             "Sudoku with deep-supervised EP.")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed (initialisation, data order, loop counts)")
    ap.add_argument("--data_dir", default=sudoku_data.RRN_DIR,
                    help="directory with train.npz and valid.npz written by sudoku.prepare "
                         "(default: data/sudoku in the repository, or $SUDOKU_RRN_DIR)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: runs/recursive/s<seed>)")
    ap.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                    help="override CONFIG keys (value parsed as JSON, else kept as a "
                         "string), e.g. --set epochs=1 bs=8")
    args = ap.parse_args()
    tag = f"recursive_s{args.seed}"
    out = args.out or os.path.join("runs", "recursive", f"s{args.seed}")

    over = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in CONFIG:
            raise SystemExit(f"--set: unknown key {k!r}")
        try:
            over[k] = json.loads(v)
        except ValueError:
            over[k] = v
    cfg = pipeline.merge_cfg(dict(CONFIG, **over, seed=args.seed))
    pipeline.set_matmul_precision(cfg)          # before any jax computation
    model.configure(cfg)

    tr_puzzles, tr_solutions = sudoku_data.load_split("train", args.data_dir)
    va_puzzles, va_solutions = sudoku_data.load_split("valid", args.data_dir)

    # solutions stay in 1..9; train() takes the first cfg["n_valid"] valid rows
    augment = functools.partial(sudoku_data.augment_batch, N=cfg["N"])
    path = pipeline.train(cfg, tr_puzzles, tr_solutions, va_puzzles, va_solutions,
                          outdir=out, tag=tag, augment_fn=augment,
                          log=lambda s: print(s, flush=True))
    print(path, flush=True)


if __name__ == "__main__":
    main()
