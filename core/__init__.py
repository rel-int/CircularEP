"""Model-agnostic machinery for Circular Predictive Coding models trained with
Equilibrium Propagation. Nothing in core/ imports a model: every function takes the model
*module* as its first argument and calls a fixed set of names on it.

A model (vision/model.py, sudoku/model.py) is a circular predictive-coding network: one
recurrent block whose state z is a tuple of value nodes, one after each sub-layer of the
block, closed into a loop at the main value node s = z[-1], which receives the input
injection and feeds the readout. A fixed point of the block sweep is the equilibrium of
the corresponding deep equilibrium model (zero prediction-error energy).

Model hooks and the core functions that call them (a model implements the hooks of the
functions it is used with):

    embed(theta, x) -> x_inj          input stem, outside the loop
                                      (ep.ep_eval, ep.coupled_theta_grad,
                                      diagnostics.spectral_radius)
    init_state(theta, x_inj) -> z     tuple of zero value nodes (ep.free_phase)
    block_sweep(theta, z, x_inj)      one Gauss-Seidel sweep over the value nodes; its
                                      fixed point is the free equilibrium
                                      (ep.free_phase, ep.free_phase_mask)
    accuracy(psi, z, y)               evaluation metric (ep.ep_eval)
    cost_sum(psi, z, y)               batch-summed cost C whose gradient nudges the
                                      state, so samples stay independent
                                      (ep.nudged_phase)
    ring(theta, x_inj) -> fns         per-node prediction maps: fns[i](z[i-1]) predicts
                                      node i from its predecessor, node 0 from z[-1]
                                      (ep.nudged_phase)
    predict(theta, z, x_inj) -> z_hat the same predictions of all nodes from the current
                                      z in one call (ep.coupled_theta_grad)
    s_map(theta, x_inj) -> f          the composed s -> s map of one sweep
                                      (diagnostics.spectral_radius)
    reduce(z) -> s, lift(s, z) -> z   optional; default s = z[-1]
                                      (ep.free_phase with solver="anderson")

Modules:
    nn           in-loop activations selected by name ("gelu", "silu@1.34", ...)
    solvers      fixed-point solvers: plain sweeps and Anderson acceleration
    ep           free phase, nudged phase, EP weight gradient, convergence gate,
                 evaluation
    diagnostics  spectral radius of the loop Jacobian
    train        learning-rate schedule and chunked accuracy evaluation
"""
