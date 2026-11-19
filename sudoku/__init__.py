"""The standard Sudoku model: a circular predictive-coding MLP-Mixer (C=512) trained with
EP on the "sudoku-hard" benchmark of Palm et al. 2018.

    python -m sudoku.prepare                                  # download + convert the data
    python -m sudoku.train [--seed S]                         # train -> runs/sudoku/s<seed>
    python -m sudoku.evaluate --ckpt runs/sudoku/s0/best_eq.pkl --views 64 --split test
"""
