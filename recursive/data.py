"""Digit-relabelling test-time views of Sudoku boards, used by recursive.evaluate.

The split files, their loader (sudoku.data.load_split, default directory
sudoku.data.RRN_DIR) and the training-time symmetry augmentation
(sudoku.data.augment_batch) are shared with the standard Sudoku model. A board is a
length-81 row-major int vector: puzzle 0 = blank / 1..9 = clue, solution 1..9; the model
consumes 0..9 and predicts 0..8 (= digit - 1).
"""
import numpy as np


def build_perms(n_views, seed):
    """[list(range(9))] ++ (n_views-1) digit permutations, perm_k ~
    np.random.default_rng(seed+k).permutation(9), resampled from the SAME rng if
    it is the identity or equal to an earlier view's permutation. Returns
    (n_views, 9) int64, view 0 the identity."""
    perms = [list(range(9))]
    seen = {tuple(perms[0])}
    for k in range(1, n_views):
        rng = np.random.default_rng(seed + k)
        while True:
            p = rng.permutation(9)
            tp = tuple(int(v) for v in p)
            if tp not in seen:
                seen.add(tp)
                perms.append(list(tp))
                break
    return np.asarray(perms, dtype=np.int64)


def relabel(puzzles, perm):
    """puzzles (n, 81) 0..9, perm (9,) int -> (n, 81) 0..9 with every clue digit d
    (1..9) relabeled to perm[d-1]+1; blanks (0) untouched."""
    perm = np.asarray(perm)
    idx = np.clip(puzzles - 1, 0, 8)
    return np.where(puzzles > 0, perm[idx] + 1, 0).astype(np.int32)


def unrelabel_pred(pred, perm):
    """pred (..., 81) 0..8 in the frame relabeled by `perm` -> same shape, mapped
    back to the original digit frame: pred_o = argsort(perm)[pred] (the inverse
    of relabel, applied to a view's prediction before anything is scored)."""
    inv = np.argsort(np.asarray(perm, dtype=np.int64))
    return inv[np.asarray(pred, dtype=np.int64)].astype(np.int8)


def clue_counts(puzzles):
    """puzzles (n, 81) 0..9 -> (n,) int number of given (non-blank) cells."""
    return (np.asarray(puzzles) > 0).sum(axis=1).astype(np.int32)
