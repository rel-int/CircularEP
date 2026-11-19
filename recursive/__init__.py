"""The recursive circular model: a Circular Predictive Coding ring (one MLP-Mixer block)
whose answer state is carried from one free equilibrium to the next, trained on Sudoku
with deep-supervised Equilibrium Propagation.

    model        the ring and the carried-answer injection: params, energy, readout,
                 spectral projection
    solvers      the Anderson-accelerated free phase and the fused +/-beta nudged phase
    ep           the centred EP weight gradient and the one-stage supervised update
    optimizers   the optax schedule / optimiser, the per-step parameter update (per-leaf
                 learning-rate scaling + spectral cap) and the EMA of the parameters
    pipeline     the recipe (DEFAULTS), the training loop and the free-chain evaluator
    data         digit-relabelling test-time views

    train        entry point: `python -m recursive.train`
    evaluate     entry point: `python -m recursive.evaluate`

`pipeline.train` takes puzzle / solution arrays (and an optional augmentation callable)
directly; sudoku.data supplies them for Sudoku.
"""
__all__ = ["model", "solvers", "ep", "optimizers", "pipeline", "data"]
