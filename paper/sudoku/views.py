"""K digit-relabelled views of every board, with the label-free self-cost selector.

View 0 is the board itself; view k >= 1 relabels the clue digits by a fixed permutation
(distinct random permutations from --view_seed, the same for every board; a prefix of
the views never depends on --views). Each view is solved independently and its
prediction is mapped back to the original digits. For every view the script stores, at
the checkpoint iterations --ck, whether the board is solved, the fraction of correct
blank cells and the self-cost
    C_self = -(1/81) sum_p max_c log softmax(x_O,p)_c
of the readout x_O. The selection of the view with the lowest C_self (ties to the first
view; oracle = any view correct) is done by paper.sudoku.analyze.

    # Table 6: 64 views of all 18k test boards, standard model at 300 sweeps, recursive
    # model at 32 stages (free-phase budget of the checkpoint); one run per checkpoint,
    # each with its own --seed label
    python -m paper.sudoku.views --model standard --ckpt runs/sudoku/s0/best_eq.pkl --seed 0
    python -m paper.sudoku.views --model recursive --ckpt runs/recursive/s0/best.pkl --seed 0
    python -m paper.sudoku.analyze table6

Boards: the first --n_boards boards of the split (0 = all). Results go to
<out_dir>/views/s<seed>_v<a>_<b>.npz (default out_dir results/sudoku/<model>_<name>), one
file per --view_chunk views; existing files are skipped, so a killed run is resumed by
repeating the command, and the views can be split over processes with --v_start / --v_end.
"""
import argparse
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

from paper.sudoku.measure import add_model_args, load_split, make_runner


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_model_args(ap)
    ap.add_argument("--views", type=int, default=64, help="total number of views K")
    ap.add_argument("--v_start", type=int, default=0,
                    help="first view computed by this process")
    ap.add_argument("--v_end", type=int, default=None,
                    help="one past the last view computed by this process (default: --views)")
    ap.add_argument("--view_chunk", type=int, default=32, help="views per saved file")
    ap.add_argument("--view_seed", type=int, default=1000,
                    help="seed of the view permutations")
    ap.add_argument("--n_boards", type=int, default=0,
                    help="first n boards of the split (0 = all)")
    ap.add_argument("--ck", default=None,
                    help="comma list of iterations whose prediction is stored "
                         "(default: the model's checkpoints up to its length)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join("results", "sudoku", f"{args.model}_{args.name}")
    parts_dir = os.path.join(out_dir, "views")
    V, VS = args.views, args.v_start
    VE = V if args.v_end is None else args.v_end
    assert 0 <= VS < VE <= V and args.view_chunk > 0
    ranges = [(a, min(a + args.view_chunk, VE)) for a in range(VS, VE, args.view_chunk)]

    def part_path(a, b):
        return os.path.join(parts_dir, f"s{args.seed}_v{a}_{b}.npz")

    todo = [(a, b) for a, b in ranges if not os.path.exists(part_path(a, b))]
    if not todo:
        print(f"all parts of views [{VS},{VE}) of seed {args.seed} exist in {parts_dir}")
        return
    os.makedirs(parts_dir, exist_ok=True)

    # boards
    puz_all, sol_all = load_split(args.data_dir, args.split)
    idx = np.arange(len(puz_all), dtype=np.int64)
    idx = idx[:args.n_boards] if args.n_boards else idx
    N = len(idx)
    puzzles, labels = puz_all[idx], sol_all[idx] - 1
    blank = puzzles == 0

    length = args.sweeps if args.model == "standard" else args.stages
    runner = make_runner(args.model, args.ckpt, length, args.t_warm)
    ck = [int(k) for k in args.ck.split(",")] if args.ck else \
        [k for k in runner.ck_default if k <= length]
    if not ck or ck[-1] != length:
        ck.append(length)
    ck_idx = np.array([k - 1 for k in ck], dtype=np.int32)
    nck = len(ck)

    perms = runner.perms(V, args.view_seed)
    assert perms.shape == (V, 9) and (perms[0] == np.arange(9)).all()
    assert len({tuple(p) for p in perms}) == V, "views are not distinct"
    assert np.array_equal(perms[:VE], runner.perms(VE, args.view_seed)), "views depend on --views"
    print(f"{args.model} seed {args.seed}: {N} boards, views [{VS},{VE}) of {V}, "
          f"{len(todo)}/{len(ranges)} files to do, checkpoints {ck}; {runner.info()}; "
          f"devices {jax.devices()}", flush=True)

    ys = [jnp.asarray(labels[i:i + args.chunk]) for i in range(0, N, args.chunk)]
    bs = [jnp.asarray(blank[i:i + args.chunk]) for i in range(0, N, args.chunk)]
    t_all = time.time()
    for a, b in todo:
        nv = b - a
        vcor = np.zeros((nck, N, nv), bool)
        vcell = np.zeros((nck, N, nv), np.float16)
        vcost = np.zeros((nck, N, nv), np.float32)
        t0 = time.time()
        for vi, v in enumerate(range(a, b)):
            inv = jnp.asarray(np.argsort(perms[v]), dtype=jnp.int32)
            for ci, i in enumerate(range(0, N, args.chunk)):
                sl = slice(i, min(i + args.chunk, N))
                xv = jnp.asarray(runner.apply_view(puzzles[sl], perms[v]).astype(np.int32))
                bd, ce, cs = runner.run(xv, ys[ci], bs[ci], inv)
                vcor[:, sl, vi] = np.asarray(bd[ck_idx])
                vcell[:, sl, vi] = np.asarray(ce[ck_idx]).astype(np.float16)
                vcost[:, sl, vi] = np.asarray(cs[ck_idx])
        tmp = part_path(a, b) + ".tmp.npz"
        np.savez(tmp, seed=np.array(args.seed), idx=idx, ckpt_iters=np.array(ck),
                 v_start=np.array(a), v_end=np.array(b), perms=perms[a:b],
                 view_correct=vcor, view_cell=vcell, view_cost=vcost)
        os.replace(tmp, part_path(a, b))
        print(f"views {a}-{b}: {time.time() - t0:.0f}s ({(time.time() - t0) / nv:.1f} s/view, "
              f"total {time.time() - t_all:.0f}s); board acc per checkpoint {ck}: "
              + " / ".join(f"{vcor[k].mean():.4f}" for k in range(nck)), flush=True)
    print(f"done seed {args.seed} views [{VS},{VE}); wall {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
