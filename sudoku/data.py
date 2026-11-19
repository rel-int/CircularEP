"""Sudoku data: the "sudoku-hard" benchmark of Palm et al. 2018 (Recurrent Relational
Networks; 180k train / 18k valid / 18k test boards, 17-34 givens) and the symmetry-group
augmentation. The split files {train,valid,test}.npz are written by
`python -m sudoku.prepare` into RRN_DIR (below); every Sudoku entry point reads them from
there unless given --data_dir.

A board is a length-P=N*N int vector (row-major), values 0..N: 0 = empty
(puzzle only), 1..N = a digit. The model one-hots this (0..N -> N+1
categories) and concatenates the fixed row/col/box one-hots; targets are
sol-1 in 0..N-1.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

BOX = {4: (2, 2), 6: (2, 3), 9: (3, 3)}     # (box_h, box_w), box_h*box_w == N


def box_dims(N):
    """Return (box_h, box_w) for an N x N Sudoku (box_h*box_w == N)."""
    return BOX[N]


def positional_ids(N):
    """Per-cell (row_id, col_id, box_id) arrays of length P=N*N (row-major)."""
    box_h, box_w = box_dims(N)
    idx = np.arange(N * N)
    row_id = idx // N
    col_id = idx % N
    box_id = (row_id // box_h) * (N // box_w) + (col_id // box_w)
    return row_id.astype(np.int32), col_id.astype(np.int32), box_id.astype(np.int32)


# Default data directory: data/sudoku under the repository root, or $SUDOKU_RRN_DIR.
# Each split file has keys "puzzles" and "solutions" (uint8, (n, 81), row-major,
# 0 = blank), in the order of the benchmark's CSV files.
RRN_DIR = os.environ.get("SUDOKU_RRN_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sudoku"))


def load_split(split, data_dir=None):
    """<data_dir>/<split>.npz (default data_dir: RRN_DIR) -> (puzzles, solutions), int32
    numpy arrays (n, 81): puzzle 0 = blank / 1..9 = clue, solution 1..9."""
    data_dir = data_dir or RRN_DIR
    path = os.path.join(data_dir, f"{split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found: run `python -m sudoku.prepare --data_dir {data_dir}` first, "
            f"or point --data_dir / $SUDOKU_RRN_DIR at the prepared split files")
    d = np.load(path)
    puz, sol = d["puzzles"], d["solutions"]
    return (puz.reshape(puz.shape[0], -1).astype(np.int32),
            sol.reshape(sol.shape[0], -1).astype(np.int32))


def get_sudoku_rrn(split_test="valid", n_test=0, data_dir=None):
    """The training split and one evaluation split: (puz_tr, sol_tr, puz_te, sol_te),
    int32 (B, 81), puzzle 0 = blank / 1..9, solution 1..9.

    split_test: the split of the evaluation arrays, "valid" during training (the test
    split is only used for the final report) or "test". n_test: keep the first n_test
    boards of it (0 = all). data_dir: default RRN_DIR."""
    puz_tr, sol_tr = load_split("train", data_dir)
    puz_te, sol_te = load_split(split_test, data_dir)
    n = n_test or len(puz_te)
    return puz_tr, sol_tr, puz_te[:n], sol_te[:n]


# ==========================================================================
# SYMMETRY AUGMENTATION
# ==========================================================================
#
# The label-preserving symmetries of Sudoku used throughout the literature (e.g. Palm
# et al. 2018): permute the row-bands and the rows within each band, permute the
# column-stacks and the columns within each stack, relabel the N digits by a random
# bijection, and (square boxes only) transpose the whole grid. Every op is applied
# independently per sample and per batch (a fresh symmetry at every step), vectorised
# in JAX.
#
# Why these preserve validity: permuting whole bands/stacks (or the rows/cols
# WITHIN one) never moves a cell across a box boundary, so every row/col/box
# still contains each remaining digit at most once; transpose is a symmetry of
# the box grid exactly when boxes are square (box_h == box_w); digit relabeling
# is just a bijection on {1..N}. The composed board is therefore still a valid
# Sudoku of IDENTICAL difficulty (same clue pattern up to symmetry). Positional
# (row/col/box) embeddings (sudoku.model._POS_OH) are recomputed from each
# cell's GRID position, not its content, so augmented boards need no change
# anywhere else in the pipeline.


def _block_perm(key, N, block, B):
    """Per-sample permutation of 0..N-1 respecting `block`-sized blocks:
    independently permute the N/block blocks AND the `block` slots within
    each block (the "permute bands, permute rows within a band" symmetry).
    Returns (B, N) int32."""
    n_blocks = N // block
    k1, k2 = jax.random.split(key)
    block_order = jnp.argsort(jax.random.uniform(k1, (B, n_blocks)), axis=1)
    within = jnp.argsort(jax.random.uniform(k2, (B, n_blocks, block)), axis=2)
    perm = (block_order[:, :, None] * block + within).reshape(B, N)
    return perm.astype(jnp.int32)


def augment_batch(key, puz, sol, N):
    """Apply one independent random Sudoku symmetry per sample to (puz, sol),
    both (B, P) int arrays (row-major; puz: 0=empty/1..N clue, sol: 1..N).
    Returns new arrays of the same shape/dtype, still a valid unique-solution
    Sudoku pair (same symmetry applied to both, so clue positions/values and
    the solution stay consistent)."""
    box_h, box_w = box_dims(N)
    B = puz.shape[0]
    kr, kc, kt, kd = jax.random.split(key, 4)
    row_perm = _block_perm(kr, N, box_h, B)
    col_perm = _block_perm(kc, N, box_w, B)

    def permute(g):
        g = g.reshape(B, N, N)
        g = jnp.take_along_axis(g, jnp.broadcast_to(row_perm[:, :, None], (B, N, N)), axis=1)
        g = jnp.take_along_axis(g, jnp.broadcast_to(col_perm[:, None, :], (B, N, N)), axis=2)
        return g

    puz_g, sol_g = permute(puz), permute(sol)

    if box_h == box_w:                                  # transpose only valid for square boxes
        do_t = jax.random.bernoulli(kt, 0.5, (B, 1, 1))
        puz_g = jnp.where(do_t, puz_g.transpose(0, 2, 1), puz_g)
        sol_g = jnp.where(do_t, sol_g.transpose(0, 2, 1), sol_g)

    puz_f, sol_f = puz_g.reshape(B, N * N), sol_g.reshape(B, N * N)
    digit_perm = jnp.concatenate(
        [jnp.zeros((B, 1), jnp.int32),
         jnp.argsort(jax.random.uniform(kd, (B, N)), axis=1).astype(jnp.int32) + 1],
        axis=1)                                         # (B, N+1); 0 (empty) fixed to 0
    puz_f = jnp.take_along_axis(digit_perm, puz_f, axis=1)
    sol_f = jnp.take_along_axis(digit_perm, sol_f, axis=1)
    return puz_f.astype(puz.dtype), sol_f.astype(sol.dtype)
