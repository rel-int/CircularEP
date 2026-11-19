"""Recurrence diagnostic: the spectral radius of the loop Jacobian at the free equilibrium
(used by the Sudoku trainer to control the spectral cap)."""
import jax
import jax.numpy as jnp

from . import ep as E


def _nrm(v):
    """Per-sample L2-normalise a batched iterate (batched power iteration helper)."""
    B = v.shape[0]
    n = jnp.linalg.norm(v.reshape(B, -1), axis=1).reshape((B,) + (1,) * (v.ndim - 1))
    return v / (n + 1e-20)


def spectral_radius(model, theta, x, free_cfg, n_iter=60):
    """rho(J): the largest |eigenvalue| of the Jacobian of the composed s -> s map
    (model.s_map) at the free equilibrium, by batched power iteration, averaged over the
    batch. Near 1 the network is a deep implicit model, near 0 it is effectively
    feedforward. free_cfg must use solver="sweep" (plain iteration)."""
    assert free_cfg.get("solver", "sweep") == "sweep", \
        "spectral_radius needs a plain-sweep free phase, not Anderson"
    x_inj = model.embed(theta, x)
    s = E.free_phase(model, theta, x_inj, free_cfg)[-1]
    f = model.s_map(theta, x_inj)
    B = s.shape[0]
    v = _nrm(jax.random.normal(jax.random.PRNGKey(1), s.shape))
    lam = 0.0
    for _ in range(n_iter):
        _, Jv = jax.jvp(f, (s,), (v,))
        lam = jnp.linalg.norm(Jv.reshape(B, -1), axis=1) / (
            jnp.linalg.norm(v.reshape(B, -1), axis=1) + 1e-20)
        v = _nrm(Jv)
    return float(jnp.mean(lam))
