"""Equilibrium Propagation (EP) for circular predictive-coding models, generic over the
model module (see core/__init__.py for the hooks each function calls).

    free_phase(model, theta, x_inj, cfg) -> z_free
        relaxation of the energy E from z = 0 (plain sweeps, or Anderson acceleration of
        the main value node)
    nudged_phase(model, theta, psi, x_inj, y, z_free, beta, cfg) -> (z_beta, e)
        relaxation of F = E + beta * C started at z_free, with the prediction errors e
    coupled_theta_grad(model, theta, x, z_beta, e, denom, w) -> theta gradient
        the EP weight gradient read off the nudged state and its errors
    free_phase_mask(model, theta, z_free, x_inj, tol) -> (w, r)
        per-sample convergence gate of the free phase
    ep_eval(model, params, x, y, cfg) -> accuracy after a free phase

Nudged phase. With e = z - predict(z) the prediction error of every value node, a
stationary point of F satisfies e = -beta dC/dz, propagated backwards through the ring
of predictions. nudged_phase iterates this as coupled Gauss-Seidel message passing on the
pair (s = z[-1], e): a forward sweep rebuilds z from s and e, a backward sweep updates e
from the cost gradient. The readout block (s, e[-1]) is Anderson-accelerated and every
other value node and error is recomputed from it at each iterate.

Weight gradient. dE/dtheta vanishes at the free equilibrium, so the one-sided EP estimate
(dE/dtheta(z_beta) - dE/dtheta(z_free)) / beta reduces to one vector-Jacobian product of
the prediction map at z_beta with the errors e (coupled_theta_grad). The input stem is
recomputed inside that product, so it receives its own gradient. The readout psi is
trained by ordinary gradient descent on the free-phase cost (done by the trainers).
"""
import jax
import jax.numpy as jnp

from . import solvers as S

_tmap = jax.tree_util.tree_map


# =========================== (a) FREE PHASE ===========================
def free_phase(model, theta, x_inj, cfg):
    """Free relaxation of E from z = model.init_state(theta, x_inj): the forward pass
    of the equilibrium model. cfg keys:
        solver   "sweep" (default): cfg["T"] block sweeps, damped by cfg["damping"]
                 (default 1.0 = plain);
                 "anderson": Anderson acceleration of the main value node only
                 (requires reduced=True, see _free_phase_reduced) for cfg["T"]
                 iterations, history depth cfg["m"] (default 6), mixing cfg["aa_beta"]
                 (default 1.0)
        reduced  True for solver="anderson"
        fast     solver="anderson": use core.solvers.anderson(fast=True) (default
                 False); cfg["hist_dtype"] is forwarded to it
    """
    z0 = model.init_state(theta, x_inj)
    solver = cfg.get("solver", "sweep")
    if solver == "sweep":
        f = lambda z: model.block_sweep(theta, z, x_inj)
        return S.sweep(f, z0, cfg["T"], damping=cfg.get("damping", 1.0))
    if solver == "anderson":
        if not cfg.get("reduced", False):
            raise ValueError("free_phase: solver='anderson' accelerates the main value "
                             "node only; set reduced=True")
        return _free_phase_reduced(model, theta, x_inj, z0, cfg)
    raise ValueError(f"free_phase: unknown solver {solver!r}")


def _reduce(model, z):
    """The reduced state of a full z: model.reduce if defined, else the main value
    node z[-1]."""
    return model.reduce(z) if hasattr(model, "reduce") else z[-1]


def _lift(model, s, z_template):
    """A full z from a reduced state s, every other value node taken from z_template:
    model.lift if defined, else z_template with its last slot replaced by s. The other
    nodes are placeholders: block_sweep recomputes every node that precedes s in the
    sweep before reading it."""
    if hasattr(model, "lift"):
        return model.lift(s, z_template)
    return z_template[:-1] + (s,)


def _free_phase_reduced(model, theta, x_inj, z0, cfg):
    """Anderson acceleration of the reduced state s only (default z[-1]), under the map
    s -> reduce(block_sweep(lift(s, z0))); the full tuple is rebuilt from the final s by
    one more block_sweep. With cfg["fast"] the final sweep starts from the last mixed
    iterate of core.solvers.anderson(fast=True, return_x=True), otherwise from the last
    image g(x) of the default implementation."""
    s0 = _reduce(model, z0)
    shp = s0.shape
    sflat0 = s0.reshape(shp[0], -1)

    def phi(sflat):
        s = sflat.reshape(shp)
        z = _lift(model, s, z0)
        s2 = _reduce(model, model.block_sweep(theta, z, x_inj))
        return s2.reshape(sflat.shape)

    if cfg.get("fast", False):
        _, sflat = S.anderson(phi, sflat0, m=cfg.get("m", 6), max_iter=cfg["T"],
                              beta=cfg.get("aa_beta", 1.0), fast=True,
                              hist_dtype=cfg.get("hist_dtype"), return_x=True)
    else:
        sflat = S.anderson(phi, sflat0, m=cfg.get("m", 6), max_iter=cfg["T"],
                           beta=cfg.get("aa_beta", 1.0))
    s_final = sflat.reshape(shp)
    return model.block_sweep(theta, _lift(model, s_final, z0), x_inj)


# ========================== (b) NUDGED PHASE ==========================
def bcast_batch(beta, a):
    """Broadcast a scalar or per-sample (B,) `beta` over the batch dimension of a
    (B, ...) array `a`."""
    if jnp.ndim(beta) == 0:
        return beta
    return beta.reshape((-1,) + (1,) * (a.ndim - 1))


def _bshape(w, a):
    """A per-sample (B,) weight `w` broadcast over the batch dimension of `a`, cast to
    a's dtype."""
    return w.reshape((-1,) + (1,) * (a.ndim - 1)).astype(a.dtype)


def _gs_forward(fns, s, e):
    """Forward sweep rebuilding the full z from the main value node s and the errors e:
    z[i] = fns[i](z[i-1]) + e[i], each reading the just-updated z[i-1] (z[-1] := s)."""
    K = len(e)
    z = [None] * K
    prev = s
    for i in range(K):
        z[i] = fns[i](prev) + e[i]
        prev = z[i]
    return tuple(z)


def _gs_coupled_step(model, theta, psi, x_inj, y, s, e, beta):
    """One coupled Gauss-Seidel step of (s, e) towards a stationary point of
    F = E + beta * C. The forward sweep rebuilds z (_gs_forward); one gradient of
    cost_sum over the whole z gives dC/dz_i for every node (zero for nodes the cost does
    not read); the backward sweep then updates e[K-2], ..., e[0] from the previous
    e[j+1] and finally e[-1] through fns[0] from the new e[0]. Generic over the number
    of value nodes K. Returns (s', e', z) with z the state rebuilt from (s, e)."""
    fns = model.ring(theta, x_inj)
    K = len(e)
    z = _gs_forward(fns, s, e)
    gC = jax.grad(lambda zz: model.cost_sum(psi, zz, y))(z)
    e_new = [None] * K
    for idx, j in enumerate(range(K - 2, -1, -1)):
        nxt = e[j + 1] if idx == 0 else e_new[j + 1]
        _, vjp = jax.vjp(fns[j + 1], z[j])
        e_new[j] = vjp(nxt)[0] - bcast_batch(beta, gC[j]) * gC[j]
    _, vjp0 = jax.vjp(fns[0], z[-1])
    e_new[-1] = vjp0(e_new[0])[0] - bcast_batch(beta, gC[-1]) * gC[-1]
    return z[-1], tuple(e_new), z


def nudged_phase(model, theta, psi, x_inj, y, z_free, beta, cfg):
    """Nudged relaxation of F = E + beta * C started at (z_free, e = 0), by coupled
    Gauss-Seidel steps (_gs_coupled_step) Anderson-accelerated on the readout block
    (s = z[-1], e[-1]) only: every other value node and its error is recomputed by the
    Gauss-Seidel step from the current block at every iterate. cfg keys: T (number of
    coupled-step evaluations; the first m fill the history), m (Anderson depth, >= 2),
    lam (ridge, default 1e-2). `beta` is a scalar or a per-sample (B,) array.

    Inside the Anderson history the errors are scaled per sample by
    ||z_free|| / (|beta| ||dC/dz(z_free)||), which balances the magnitudes of the two
    halves of the block for the least squares; every coupled step sees the unscaled
    (s, e), so the scaling changes the mixing weights only, not the fixed point.
    Returns (z_beta, e)."""
    T, m, lam = cfg["T"], cfg["m"], cfg.get("lam", 1e-2)
    if m < 2:
        raise ValueError(f"nudged_phase: Anderson depth m must be >= 2, got {m}")
    v0, shapes = S.flatten_state(z_free)
    B, D = v0.shape
    Ds = 1
    for d in shapes[-1][1:]:
        Ds *= d

    def unf(v):
        return S.unflatten_state(v, shapes)

    gC0, _ = S.flatten_state(jax.grad(lambda zz: model.cost_sum(psi, zz, y))(z_free))
    sc = (jnp.linalg.norm(v0, axis=1)
          / (jnp.abs(beta) * jnp.linalg.norm(gC0, axis=1) + 1e-30))[:, None]

    def G(x):
        s_, e_, z_ = _gs_coupled_step(model, theta, psi, x_inj, y,
                                      unf(x[:, :D])[-1], unf(x[:, D:] / sc), beta)
        return jnp.concatenate([S.flatten_state(z_)[0], S.flatten_state(e_)[0] * sc], axis=1)

    blk = lambda v: jnp.concatenate([v[:, D - Ds:D], v[:, 2 * D - Ds:]], axis=1)
    put_blk = lambda v, b: v.at[:, D - Ds:D].set(b[:, :Ds]).at[:, 2 * D - Ds:].set(b[:, Ds:])
    x0 = jnp.concatenate([v0, jnp.zeros_like(v0)], axis=1)
    f = G(x0)
    Fh = jnp.zeros((B, m, 2 * Ds), v0.dtype).at[:, 0, :].set(blk(f))
    Gh = jnp.zeros((B, m, 2 * Ds), v0.dtype).at[:, 0, :].set(blk(f) - blk(x0))

    def warm(i, carry):
        Fh, Gh, f = carry
        fn = G(f)
        return Fh.at[:, i, :].set(blk(fn)), Gh.at[:, i, :].set(blk(fn) - blk(f)), fn

    Fh, Gh, f = jax.lax.fori_loop(1, m, warm, (Fh, Gh, f))
    gram = jnp.einsum("bmd,bnd->bmn", Gh, Gh, precision=jax.lax.Precision.HIGHEST)

    def body(k, carry):
        Fh, Gh, gram, f = carry
        a = S.mixing_weights(gram, lam)
        xb = jax.lax.optimization_barrier(jnp.sum(a[:, :, None] * Fh, axis=1))
        fn = G(put_blk(f, xb))
        gn = blk(fn) - xb
        s_ = k % m
        Fh, Gh = Fh.at[:, s_, :].set(blk(fn)), Gh.at[:, s_, :].set(gn)
        r = jnp.sum(Gh * gn[:, None, :], axis=-1)
        gram = gram.at[:, s_, :].set(r).at[:, :, s_].set(r)
        return Fh, Gh, gram, fn

    _, _, _, f = jax.lax.fori_loop(m, max(T, m + 1), body, (Fh, Gh, gram, f))
    z, e = unf(f[:, :D]), unf(f[:, D:] / sc)
    return z, e


# ======================= (c) EP WEIGHT GRADIENT =======================
def coupled_theta_grad(model, theta, x, z_beta, e, denom, w=None):
    """-VJP_theta[theta -> model.predict(theta, z_beta, model.embed(theta, x))](w * e)
    / denom: the EP weight gradient from a nudged state z_beta and its errors e. w is
    an optional per-sample (B,) weight; for a one-sided estimate with nudge beta pass
    w = 1 / beta (times any per-sample gate) and denom = the number of samples."""
    ew = e if w is None else _tmap(lambda a: a * _bshape(w, a), e)
    _, vjp = jax.vjp(lambda th: model.predict(th, z_beta, model.embed(th, x)), theta)
    return _tmap(lambda g: -g / denom, vjp(ew)[0])


# ============================ (d) CONVERGENCE GATE ============================
def free_phase_mask(model, theta, z_free, x_inj, tol):
    """Per-sample convergence gate of the free phase: w_i = 1[||sweep(z)_i - z_i|| /
    ||z_i|| < tol]. EP assumes z_free is an equilibrium; samples that have not settled
    give a meaningless contrastive estimate, so the trainer can drop them. Returns
    (w, r): the 0/1 mask and the per-sample relative residual."""
    z1 = model.block_sweep(theta, z_free, x_inj)
    d, _ = S.flatten_state(_tmap(lambda a, b: a - b, z1, z_free))
    v, _ = S.flatten_state(z_free)
    r = jnp.linalg.norm(d, axis=1) / (jnp.linalg.norm(v, axis=1) + 1e-20)
    return (r < tol).astype(v.dtype), r


# =============================== (e) EVAL ===============================
def ep_eval(model, params, x, y, cfg):
    """Accuracy after a free phase with `cfg` (see free_phase), e.g. {"solver": "sweep",
    "T": 40} or {"solver": "anderson", "reduced": True, "m": 9, "T": 40}."""
    theta = params["theta"]
    z = free_phase(model, theta, model.embed(theta, x), cfg)
    return model.accuracy(params["psi"], z, y)
