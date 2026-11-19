"""Relaxations of the recursive ring: the Anderson-accelerated free phase, the fused
+/-beta heavy-ball nudged phase and the free-state residual. Anderson acceleration and
the tuple <-> flat-array helpers are core.solvers' (default implementation).

The free phase is always Anderson acceleration run to its full iteration budget; the
nudged phase is a fixed number of heavy-ball steps.
"""
import jax
import jax.numpy as jnp

from core.solvers import anderson, flatten_state, unflatten_state
from recursive import model as M

_tmap = jax.tree_util.tree_map


def relax_free(theta, x_inj, z_warm, *, cold, cfg):
    """Free phase: Anderson(m = cfg["m"]) acceleration of the block sweep, for
    cfg["T_free"] iterations from zero (cold, the datapoint's single cold start)
    or cfg["T_free_warm"] iterations from z_warm (warm)."""
    T = cfg["T_free"] if cold else cfg["T_free_warm"]
    if cold:
        z_warm = M.init_state(theta, x_inj)
    v0, shapes = flatten_state(z_warm)

    def g(v):
        z = unflatten_state(v, shapes)
        vf, _ = flatten_state(M.block_sweep(theta, z, x_inj))
        return vf

    return unflatten_state(anderson(g, v0, m=cfg["m"], max_iter=T), shapes)


def relax_nudged_pm(theta, psi, x_inj, z_free, y_lab, cfg):
    """The fused +/-beta nudged pair: ONE 2B-row heavy-ball relaxation of
    F = E + beta_i*C (beta_i = +cfg["beta"] on the first B rows, -cfg["beta"] on
    the next B), both halves warm-started at z_free, for cfg["T_nudg"] steps with
    (lr_h, mom) = (cfg["lr_h"], cfg["mom"]). gradC only touches the readout node,
    so splitting grad(F) into grad(E) + beta*grad(C) costs ~nothing over the
    fused gradient, and running the +/- pair as one 2B-row relaxation costs
    ~1 nudge instead of 2 in wall-clock.

    `psi` is required to form grad(C) (the nudge cost reads the readout weights);
    it is not part of `theta` (the CONTRAST's theta-grad, computed separately,
    never needs psi)."""
    beta, T, lr_h, mom = cfg["beta"], cfg["T_nudg"], cfg["lr_h"], cfg["mom"]
    B = x_inj.shape[0]
    tile2 = lambda a: jnp.concatenate([a, a], axis=0)
    bvec = jnp.concatenate([jnp.full((B,), beta, x_inj.dtype),
                            jnp.full((B,), -beta, x_inj.dtype)])
    x2, y2 = tile2(x_inj), tile2(y_lab)
    z2 = _tmap(tile2, z_free)
    gradE = jax.grad(lambda zz: M.energy_internal(theta, zz, x2))
    gradC = jax.grad(lambda zz: M.cost_sum(psi, zz, y2))
    v0, shapes = flatten_state(z2)

    def g(v):
        zz = unflatten_state(v, shapes)
        ge, _ = flatten_state(gradE(zz))
        gc, _ = flatten_state(gradC(zz))
        return ge + bvec[:, None] * gc

    def body(_, carry):
        v, mo = carry
        mo = mom * mo - lr_h * g(v)
        return (v + mo, mo)

    v, _ = jax.lax.fori_loop(0, T, body, (v0, jnp.zeros_like(v0)))
    z_n = unflatten_state(v, shapes)
    zp = _tmap(lambda a: a[:B], z_n)
    zm = _tmap(lambda a: a[B:], z_n)
    return zp, zm


def free_residual(theta, x_inj, z):
    """Per-sample relative one-sweep residual ||sweep(z) - z|| / ||z|| of a free
    state: EP assumes z is an equilibrium, and recursive.evaluate counts a free phase
    as converged when this is below 1e-3."""
    z1 = M.block_sweep(theta, z, x_inj)
    d, _ = flatten_state(_tmap(lambda a, b: a - b, z1, z))
    v, _ = flatten_state(z)
    return jnp.linalg.norm(d, axis=1) / (jnp.linalg.norm(v, axis=1) + 1e-20)
