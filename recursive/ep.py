"""Equilibrium Propagation: the centred contrastive theta-grad and the one-stage
supervised update (free relaxation -> fused nudged pair -> contrast + local
grads -> the carried answer update).

The theta-grad g(z) = d/dtheta E(theta, z, embed(theta, puz, y)) recomputes the
stem's injection INSIDE the grad, so the stem receives its own gradient
(holding x_inj fixed would leave the embedding untrained). For the ring matrices
g(z) is a local (postsynaptic residual) x (presynaptic activity) product; theta
is trained by this contrastive rule, psi (the readout) by ordinary gradient
descent of the free-phase cost.
"""
import jax
import jax.numpy as jnp

from recursive import model as M
from recursive import solvers as S

_tmap = jax.tree_util.tree_map


def _energy_theta_grad(theta, puz, y, z):
    """g(z) = d/dtheta E(theta, z, embed(theta, puz, y))."""
    return jax.grad(lambda th: M.energy_internal(th, z, M.embed(th, puz, y)))(theta)


def contrastive_theta_grad(theta, x, z_free, z_plus, z_minus, beta):
    """Centred EP estimator of dC/dtheta: (g(z_plus) - g(z_minus)) / (2 beta B).
    x = (puz, y), the stage's embed input; B = the number of contrasted rows.
    z_free is accepted for a uniform estimator signature; the centred formula
    does not read it."""
    del z_free
    puz, y = x
    B = z_plus[-1].shape[0]
    g = lambda z: _energy_theta_grad(theta, puz, y, z)
    scale = 1.0 / (2.0 * beta * B)
    return _tmap(lambda u, v: (u - v) * scale, g(z_plus), g(z_minus))


def local_grads(params, z_free, x_inj, y_lab, cfg):
    """The stage's local terms at the fixed free equilibrium: the var-penalty
    theta-grad (cfg["var_pen"] <= 0 -> None, no-op) and the delta-rule psi-grad
    of the free-phase CE. `x_inj` (the stem injection that produced z_free) is
    required by the var penalty, which reads theta-dependent predictions at
    fixed z; it is not part of z_free itself."""
    theta, psi = params["theta"], params["psi"]
    var_pen, var_target = cfg["var_pen"], cfg["var_target"]
    theta_pen_grad = None
    if var_pen > 0.0:
        theta_pen_grad = jax.grad(
            lambda th: var_pen * M.var_penalty_pred(th, z_free, x_inj, var_target))(theta)
    psi_grad = jax.grad(lambda ps: M.ce_loss(ps, z_free, y_lab))(psi)
    return theta_pen_grad, psi_grad


def _prune_select(psi, y_carry, y_lab, key, K, keep_frac):
    """The K rows of a loop that the nudged phase + contrast see, chosen from the
    previous loop's answer: pred = argmax readout(psi, y_carry), solved = every
    cell correct. With n_keep = round(keep_frac K) and the rows in a random
    order: an unsolved row is always selected first; a solved row is selected
    only up to n_keep slots, the rest only once the unsolved rows run out."""
    B = y_lab.shape[0]
    pred = jnp.argmax(M.readout(psi, y_carry), axis=-1)            # (B, P)
    solved = jnp.all(pred == y_lab, axis=1)                        # (B,)
    n_keep = int(round(keep_frac * K))
    pos = jax.random.permutation(key, B)
    u_pos = jnp.zeros((B,), jnp.int32).at[pos].set((~solved).astype(jnp.int32))
    s_pos = 1 - u_pos
    rank_u = (jnp.cumsum(u_pos) - u_pos)[pos]
    rank_s = (jnp.cumsum(s_pos) - s_pos)[pos]
    keys = jnp.where(solved, jnp.where(rank_s < n_keep, rank_s - n_keep, K + rank_s), rank_u)
    return jnp.argsort(keys)[:K]


def stage_grads(params, x, y_lab, y_carry, z_warm, key, *, cold, prune_k, cfg):
    """One supervised energy minimisation: free relaxation (cold or warm) -> the
    nudged pair + contrastive theta-grad (on all B rows, or on prune_k rows
    gathered from the previous loop's answer when 0 < prune_k < B) -> the
    var-penalty + delta-rule local grads (full batch) -> the carried answer
    update. Returns ({"theta": theta_grad, "psi": psi_grad}, z_free, y_new,
    logs) with logs e_free, loss, prune_rows (rows contrasted) and dy (relative answer
    change)."""
    theta, psi = params["theta"], params["psi"]
    x_inj = M.embed(theta, x, y_carry)
    z_free = S.relax_free(theta, x_inj, z_warm, cold=cold, cfg=cfg)
    B = x_inj.shape[0]
    logs = {"e_free": M.energy_internal(theta, z_free, x_inj) / B,
            "loss": M.ce_loss(psi, z_free, y_lab)}

    rows = None
    if 0 < prune_k < B:
        key, ksel = jax.random.split(key)
        rows = _prune_select(psi, y_carry, y_lab, ksel, prune_k, cfg["prune_keep"])
    logs["prune_rows"] = jnp.asarray(float(B if rows is None else prune_k), jnp.float32)

    if rows is None:
        xr, ycr, yr, xir, zr = x, y_carry, y_lab, x_inj, z_free
    else:
        take = lambda a: a[rows]
        xr, ycr, yr, xir = take(x), take(y_carry), take(y_lab), take(x_inj)
        zr = _tmap(take, z_free)
    zp, zm = S.relax_nudged_pm(theta, psi, xir, zr, yr, cfg)
    theta_grad = contrastive_theta_grad(theta, (xr, ycr), zr, zp, zm, cfg["beta"])

    pen_grad, psi_grad = local_grads(params, z_free, x_inj, y_lab, cfg)
    if pen_grad is not None:
        theta_grad = _tmap(lambda g, pg: g + pg, theta_grad, pen_grad)

    y_new = M.answer_update(y_carry, z_free[-1], cfg["y_rate"])
    logs["dy"] = M.rel_change(y_new, y_carry)
    return {"theta": theta_grad, "psi": psi_grad}, z_free, y_new, logs
