"""Fixed-point solvers for the free and nudged phases, and the helpers that pack a tuple
state into the per-sample (B, D) array that Anderson acceleration works on.

    sweep     plain (optionally damped) iteration of a map. With the block sweep of a
              model this is the physical dynamics of the network: every value node is
              updated from its own local prediction.
    anderson  Anderson acceleration of x = g(x): a per-sample least-squares mixing of the
              last m iterates. It needs a reduction over the whole state, so it is a
              simulation device; it converges to the same fixed point as plain iteration.

`anderson` has two implementations of the same method. The default keeps the iterates X
and their images F and recomputes the Gram matrix of the residuals F - X at every step.
`fast=True` keeps F and the residuals G, updates one row and column of the Gram matrix per
step (cheaper for long histories) and can store G in a lower precision (`hist_dtype`).
Both converge to the same fixed point, but their iterates differ in floating point, which
matters when a phase is stopped after a few iterations: the Sudoku recipe uses
`fast=True`, the vision recipes the default.
"""
import jax
import jax.numpy as jnp
from jax import lax

_tmap = jax.tree_util.tree_map


# --------------------------------- sweep ---------------------------------
def sweep(f, z0, T, damping=1.0):
    """T applications of the fixed-point map f from z0, optionally damped:
    z <- (1 - damping) z + damping f(z). damping=1.0 is plain iteration z <- f(z)."""
    if damping == 1.0:
        return jax.lax.fori_loop(0, T, lambda _, z: f(z), z0)

    def body(_, z):
        fz = f(z)
        return _tmap(lambda a, b: (1.0 - damping) * a + damping * b, z, fz)

    return jax.lax.fori_loop(0, T, body, z0)


# -------------------------------- anderson --------------------------------
def mixing_weights(gram, lam):
    """Anderson mixing weights alpha (B, m) from a Gram matrix (B, m, m) of residuals:
    the ridge-regularised least squares min ||sum_i alpha_i G_i|| subject to
    sum_i alpha_i = 1, solved as a KKT system. The ridge lam is relative to the mean
    diagonal of the Gram matrix."""
    B, m, _ = gram.shape
    scale = jnp.mean(jnp.diagonal(gram, axis1=1, axis2=2), axis=1)[:, None, None]
    A = gram + lam * (scale + 1e-12) * jnp.eye(m, dtype=gram.dtype)[None]
    H = jnp.zeros((B, m + 1, m + 1), gram.dtype)       # KKT system
    H = H.at[:, 0, 1:].set(1.0).at[:, 1:, 0].set(1.0).at[:, 1:, 1:].set(A)
    e0 = jnp.zeros((B, m + 1, 1), gram.dtype).at[:, 0, 0].set(1.0)
    return jnp.linalg.solve(H, e0)[:, 1:, 0]


def _mix(alpha, F, G, beta):
    """x = beta sum a F + (1-beta) sum a X = sum a F - (1-beta) sum a G (X = F - G)."""
    x = jnp.sum(alpha[:, :, None] * F, axis=1)
    if beta != 1.0:
        x = x - (1.0 - beta) * jnp.sum(alpha[:, :, None] * G.astype(alpha.dtype), axis=1)
    return x


def _anderson_fast(g, x0, m, max_iter, lam, beta, hist_dtype, return_x):
    """The `fast=True` implementation of `anderson`. Returns g(x) of the last mixed
    iterate x (and x itself with return_x=True)."""
    B, D = x0.shape
    hd = x0.dtype if hist_dtype is None else hist_dtype
    put = lambda H, i, v: lax.dynamic_update_index_in_dim(H, v.astype(H.dtype), i, 1)
    f = g(x0)
    F = put(jnp.zeros((B, m, D), x0.dtype), 0, f)
    G = put(jnp.zeros((B, m, D), hd), 0, f - x0)

    def warm(i, carry):
        """Seed slot i with the next plain iterate x = g(previous x)."""
        F, G, f = carry
        fn = g(f)
        return put(F, i, fn), put(G, i, fn - f), fn

    F, G, f = lax.fori_loop(1, m, warm, (F, G, f))
    Gf = G.astype(x0.dtype)
    gram = jnp.einsum("bmd,bnd->bmn", Gf, Gf, precision=lax.Precision.HIGHEST)

    def body(k, carry):
        """One Anderson step; only slot k % m and its gram row/col change."""
        F, G, gram, x, f = carry
        x = lax.optimization_barrier(_mix(mixing_weights(gram, lam), F, G, beta))
        f = g(x)
        gn = f - x
        s = k % m
        F, G = put(F, s, f), put(G, s, gn)
        r = jnp.sum(G.astype(x0.dtype) * gn[:, None, :], axis=-1)     # (B, m)
        gram = lax.dynamic_update_index_in_dim(gram, r, s, 1)
        gram = lax.dynamic_update_index_in_dim(gram, r, s, 2)
        return F, G, gram, x, f

    _, _, _, x, f = lax.fori_loop(m, max(max_iter, m + 1), body, (F, G, gram, f, f))
    return (f, x) if return_x else f


def anderson(g, x0, m=6, max_iter=40, lam=1e-2, beta=1.0, *, fast=False,
             hist_dtype=None, return_x=False):
    """Per-sample Anderson acceleration solving x = g(x), with x0 and g(x) of shape
    (B, D). The depth-m cyclic history is seeded by m plain iterations of g; every
    further step mixes the history with the weights of `mixing_weights` (ridge `lam`,
    relative to the Gram scale). `beta` is the mixing parameter (not the EP nudge
    strength). Runs max(max_iter, m + 1) evaluations of g in total and returns the
    last one.

    fast=True selects the second implementation (see the module docstring);
    `hist_dtype` (store the residual history in e.g. bfloat16; default: x0's dtype) and
    `return_x` (also return the last mixed iterate x, not only g(x)) apply to it only."""
    if fast:
        return _anderson_fast(g, x0, m, max_iter, lam, beta, hist_dtype, return_x)
    assert not return_x, "return_x is only meaningful with fast=True"
    B, D = x0.shape
    e0 = jnp.zeros((B, m + 1, 1), x0.dtype).at[:, 0, 0].set(1.0)
    eye = jnp.eye(m, dtype=x0.dtype)[None]
    X = jnp.zeros((B, m, D), x0.dtype).at[:, 0, :].set(x0)
    F = jnp.zeros((B, m, D), x0.dtype).at[:, 0, :].set(g(x0))

    def warm(i, carry):
        """Seed history slot i with the next plain iterate of the previous one."""
        X, F = carry
        xi = F[:, i - 1, :]
        X = X.at[:, i, :].set(xi)
        F = F.at[:, i, :].set(g(xi))
        return (X, F)

    X, F = jax.lax.fori_loop(1, m, warm, (X, F))

    def body(k, carry):
        """One Anderson step: solve the constrained LS for the mixing weights alpha,
        form the accelerated iterate, and overwrite the cyclic history slot."""
        X, F = carry
        G = F - X                                      # (B, m, D) residuals
        gram = jnp.einsum("bmd,bnd->bmn", G, G)
        scale = jnp.mean(jnp.einsum("bmd,bmd->bm", G, G), axis=1)[:, None, None]
        H = jnp.zeros((B, m + 1, m + 1), x0.dtype)
        H = H.at[:, 0, 1:].set(1.0).at[:, 1:, 0].set(1.0)
        H = H.at[:, 1:, 1:].set(gram + lam * (scale + 1e-12) * eye)
        alpha = jnp.linalg.solve(H, e0)[:, 1:, 0]      # (B, m)
        xk = beta * jnp.einsum("bm,bmd->bd", alpha, F) \
            + (1 - beta) * jnp.einsum("bm,bmd->bd", alpha, X)
        slot = k % m
        X = X.at[:, slot, :].set(xk)
        F = F.at[:, slot, :].set(g(xk))
        return (X, F)

    X, F = jax.lax.fori_loop(m, jnp.maximum(max_iter, m + 1), body, (X, F))
    last = (jnp.maximum(max_iter, m + 1) - 1) % m
    return F[jnp.arange(B), last, :]


# --------------------------- pytree <-> flat array ---------------------------
def flatten_state(z):
    """Pack a tuple state z = (v_1, ..., v_k) of (B, ...) arrays into a flat
    per-sample (B, D) array for Anderson; returns (v, shapes) for unflatten_state."""
    B = z[0].shape[0]
    v = jnp.concatenate([leaf.reshape(B, -1) for leaf in z], axis=1)
    return v, tuple(leaf.shape for leaf in z)


def unflatten_state(v, shapes):
    """Inverse of flatten_state: split a flat (B, D) array back into the tuple."""
    out, off = [], 0
    for sh in shapes:
        sz = 1
        for d in sh[1:]:
            sz *= d
        out.append(v[:, off:off + sz].reshape(sh))
        off += sz
    return tuple(out)
