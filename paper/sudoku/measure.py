"""Single-view Sudoku measurement: the prediction after every sweep (standard model) or
stage (recursive model), and the model runners shared with paper.sudoku.views.

    # standard model: damped readout (a = 0.5), prediction after each of --sweeps sweeps
    python -m paper.sudoku.measure --model standard --ckpt runs/sudoku/s0/best_eq.pkl \\
        [--sweeps 300]
    # recursive model: prediction after each of --stages stages (free-phase budget of the
    # checkpoint, or --t_warm for the warm stages)
    python -m paper.sudoku.measure --model recursive --ckpt runs/recursive/s0/best.pkl \\
        [--stages 32]

Writes <out_dir>/meas_s<seed>.npz (default out_dir results/sudoku/<model>_<name>) with,
for every board of the split (the 18k test boards by default):
    board (iters, B) bool   all 81 cells correct
    cell  (iters, B) f32    fraction of correct blank cells
    solve_iter (B,) int16   first iteration from which the board stays correct up to the
                            last one; 0 = not correct at the last iteration
"""
import argparse
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

from sudoku import data as sudoku_data

CK_STANDARD = (10, 25, 30, 50, 75, 100, 300, 500, 1000, 1500)
CK_RECURSIVE = (1, 2, 4, 8, 16, 32)


def solve_iteration(board):
    """board (T, B) bool -> (B,) int16: first iteration from which the board stays correct
    up to the last one, 0 = not correct at the last iteration."""
    stay = np.logical_and.accumulate(board[::-1], axis=0)[::-1]      # stay[t] = all(board[t:])
    first = stay.argmax(0) + 1
    first[~board[-1]] = 0
    return first.astype(np.int16)


def load_split(data_dir, split="test", n=0):
    """(puzzles, solutions) int32 (B, 81) of <data_dir>/<split>.npz: 0 = blank, digits
    1..9; the first n boards only if n > 0."""
    puz, sol = sudoku_data.load_split(split, data_dir)
    assert puz.shape == sol.shape and puz.shape[1] == 81, puz.shape
    return (puz[:n], sol[:n]) if n else (puz, sol)


class Standard:
    """Standard model, damped readout: s <- (1 - a) s + a * sweep(z)[-1] (the other value
    nodes take the fresh sweep value), read out after every sweep."""
    kind = "standard"
    ck_default = CK_STANDARD

    def __init__(self, ckpt, length, a=0.5):
        jax.config.update("jax_default_matmul_precision", "tensorfloat32")
        from sudoku import evaluate as ev
        from sudoku import model as M
        params, _, readouts = ev.load_checkpoint(ckpt)
        self.theta, self.psi = params["theta"], params["psi"]
        self.length, self.readouts = length, readouts
        self.perms = ev.view_perms
        self.apply_view = ev.apply_view
        T, A = length, a

        @jax.jit
        def _run(theta, psi, X, Y, blank, inv):
            x_inj = M.embed(theta, X)
            z0 = M.init_state(theta, x_inj)
            nb = jnp.maximum(blank.sum(axis=1), 1).astype(jnp.float32)

            def body(z, _):
                zs = M.block_sweep(theta, z, x_inj)
                z = zs[:-1] + ((1.0 - A) * z[-1] + A * zs[-1],)
                logits = M.readout(psi, z[-1])
                corr = inv[jnp.argmax(logits, axis=-1)] == Y
                c_self = -jnp.mean(jnp.max(jax.nn.log_softmax(logits, axis=-1), axis=-1), axis=1)
                return z, (corr.all(axis=1), ((corr & blank).sum(axis=1) / nb).astype(jnp.float32),
                           c_self.astype(jnp.float32))

            return jax.lax.scan(body, z0, None, length=T)[1]

        self._run = _run

    def run(self, X, Y, blank, inv):
        """X (b,81) int32 puzzle in the view's digit frame, Y (b,81) labels 0..8 in the
        original frame, blank (b,81) bool, inv (9,) = argsort(view permutation) ->
        board (iters,b) bool, cell (iters,b) f32, C_self (iters,b) f32."""
        return self._run(self.theta, self.psi, X, Y, blank, inv)

    def info(self):
        """Settings stored with the outputs."""
        return dict(model=self.kind, iters=self.length)


class Recursive:
    """Recursive model: the free chain of `length` stages (cold first stage, then warm)."""
    kind = "recursive"
    ck_default = CK_RECURSIVE

    def __init__(self, ckpt, length, t_warm=None):
        from recursive import model, pipeline
        from recursive.data import build_perms, relabel
        params, cfg = pipeline.load_checkpoint(ckpt)
        if t_warm is not None:
            cfg["T_free_warm"] = int(t_warm)
        pipeline.set_matmul_precision(cfg)
        model.configure(cfg)
        self.params, self.cfg, self.length, self._chain = params, cfg, length, pipeline.chain
        self.perms, self.apply_view = build_perms, relabel
        self.total = cfg["T_free"] + (length - 1) * cfg["T_free_warm"]

        @jax.jit
        def _post(probs, C, lab, blk, inv):
            pred_o = inv[jnp.argmax(probs, axis=-1)]                      # (S,b,81)
            corr = pred_o == lab[None]
            nb = jnp.maximum(blk.sum(axis=1), 1).astype(jnp.float32)
            cell = ((corr & blk[None]).sum(axis=2) / nb[None]).astype(jnp.float32)
            return corr.all(axis=2), cell, C.astype(jnp.float32)

        self._post = _post

    def run(self, X, Y, blank, inv):
        """As Standard.run; iterations are stages and C_self is that of each stage."""
        out = self._chain(self.params, X, self.length, self.cfg)
        return self._post(out["probs"], out["C_self"], Y, blank, inv)

    def info(self):
        """Settings stored with the outputs."""
        return dict(model=self.kind, iters=self.length, t_free=self.cfg["T_free"],
                    t_warm=self.cfg["T_free_warm"], total_iters=self.total)


def make_runner(kind, ckpt, length, t_warm=None):
    """Standard(ckpt, length) or Recursive(ckpt, length, t_warm)."""
    return Standard(ckpt, length) if kind == "standard" else Recursive(ckpt, length, t_warm)


def add_model_args(ap):
    """The model / data arguments shared by measure and views."""
    ap.add_argument("--model", required=True, choices=["standard", "recursive"],
                    help="standard (sudoku.train) or recursive (recursive.train) model")
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint: best_eq.pkl of sudoku.train / best.pkl of "
                         "recursive.train")
    ap.add_argument("--seed", type=int, default=0,
                    help="label of the checkpoint, names the output file; give each checkpoint "
                         "of the same --name a distinct label")
    ap.add_argument("--sweeps", type=int, default=300, help="standard model: sweeps")
    ap.add_argument("--stages", type=int, default=32, help="recursive model: stages")
    ap.add_argument("--t_warm", type=int, default=None,
                    help="recursive model: free-phase steps of the warm stages "
                         "(default: the checkpoint's)")
    ap.add_argument("--name", default="table6",
                    help="run name, output folder results/sudoku/<model>_<name>")
    ap.add_argument("--out_dir", default=None,
                    help="output folder (default: results/sudoku/<model>_<name>)")
    ap.add_argument("--data_dir", default=sudoku_data.RRN_DIR,
                    help="folder with <split>.npz written by sudoku.prepare (default: "
                         "data/sudoku in the repository, or $SUDOKU_RRN_DIR)")
    ap.add_argument("--split", default="test", choices=["valid", "test"],
                    help="split to evaluate on")
    ap.add_argument("--chunk", type=int, default=1000, help="boards per jitted call")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_model_args(ap)
    ap.add_argument("--n_boards", type=int, default=0, help="first n boards only (0 = all)")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join("results", "sudoku", f"{args.model}_{args.name}")
    out = os.path.join(out_dir, f"meas_s{args.seed}.npz")
    if os.path.exists(out):
        print(f"{out} exists; nothing to do")
        return

    puzzles, solutions = load_split(args.data_dir, args.split, args.n_boards)
    B = puzzles.shape[0]
    labels, blank = solutions - 1, puzzles == 0
    length = args.sweeps if args.model == "standard" else args.stages
    runner = make_runner(args.model, args.ckpt, length, args.t_warm)
    print(f"{args.model} seed {args.seed}: {B} boards, {length} iterations; {runner.info()}; "
          f"devices {jax.devices()}", flush=True)

    ident = jnp.arange(9, dtype=jnp.int32)
    board = np.zeros((length, B), bool)
    cell = np.zeros((length, B), np.float32)
    t0 = time.time()
    for i in range(0, B, args.chunk):
        sl = slice(i, min(i + args.chunk, B))
        bd, ce, _ = runner.run(jnp.asarray(puzzles[sl]), jnp.asarray(labels[sl]),
                               jnp.asarray(blank[sl]), ident)
        board[:, sl], cell[:, sl] = np.asarray(bd), np.asarray(ce)
        print(f"chunk {i}-{sl.stop} ({time.time() - t0:.0f}s): board acc first/last iteration "
              f"{board[0, :sl.stop].mean():.4f} / {board[-1, :sl.stop].mean():.4f}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    tmp = out + ".tmp.npz"
    np.savez(tmp, seed=np.array(args.seed), board=board, cell=cell,
             solve_iter=solve_iteration(board), puzzles=puzzles.astype(np.uint8),
             n_given=(puzzles > 0).sum(1).astype(np.int32),
             **{k: np.array(v) for k, v in runner.info().items()})
    os.replace(tmp, out)
    print("saved", out, f"wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
