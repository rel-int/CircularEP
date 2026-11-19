"""The standard Sudoku model: an attention-free MLP-Mixer block run to a fixed point, as a
circular predictive-coding network with four value nodes. It implements the model hooks
of core/__init__.py used by core.ep and core.diagnostics (embed, init_state, block_sweep,
accuracy, cost_sum, ring, predict, s_map) plus the energy, the readout and the spectral
projection used by sudoku/train.py and sudoku/evaluate.py.

The block is one pre-norm MLP-Mixer layer (a token-mixing MLP in place of self-attention,
a channel-mixing MLP as the feed-forward part) with one value node per linear sub-layer,
closed into a loop at the main node s, which is also the readout node:

    state z = (h_t, t, h_c, s)
      h_t : (B, C, D_t)   token-mix hidden     h_t_hat(s)    = act(LN(s)^T @ Wt1)
      t   : (B, P, C)     token-mix output     t_hat(h_t)    = (h_t @ Wt2)^T
      h_c : (B, P, D_c)   channel-mix hidden   h_c_hat(t)    = act(LN(t) @ Wc1)
      s   : (B, P, C)     channel-mix output   s_hat(h_c, x) = h_c @ Wc2 + x_inj

    E(z) = 1/2 sum of the 4 squared prediction residuals; readout = per-cell affine LN +
    linear.

The P = N*N cells are the tokens; the channel dimension C carries the per-cell digit
belief. Token mixing propagates constraints across cells (a linear map, hence spectrally
cappable: the contractivity knob); channel mixing does per-cell digit reasoning. There is
no identity residual (it would break contractivity). The stem (clue one-hot + fixed
row/column/box one-hots -> x_inj) is feedforward and recomputed inside the EP weight
gradient, so it receives its own gradient. LN is a non-affine LayerNorm with variance
floor NORM_EPS.

configure(cfg) sets the module globals from a recipe dict before init_params / embed.
"""
import jax
import jax.numpy as jnp
import numpy as np

from core.nn import apply_act
from . import data as _data

# -------- geometry / capacity (set by configure) --------
N = 4
P = 16
D_T = 16            # token-mix hidden width
D_C = 256           # channel-mix hidden width
N_CLASSES = 4
_POS_OH = None      # fixed positional one-hots (P, 3N), constant w.r.t. theta
F_IN = 0            # stem input feature dim = (N+1) + 3N

# In-loop activation (both mixing MLPs), core.nn name, optionally "base@t".
ACT = "gelu"

# In-loop LayerNorm variance floor: the LN gain is 1/sqrt(var + NORM_EPS), so NORM_EPS
# bounds it by 1/sqrt(NORM_EPS) (this gain is what can push rho(J) past the spectral cap).
NORM_EPS = 1e-5

TOK_LEAVES = ("Wt1", "bt1", "Wt2", "bt2")     # per-layer lr groups (sudoku/train.py)
CHAN_LEAVES = ("Wc1", "bc1", "Wc2", "bc2")
MIX_MATS = ("Wt1", "Wt2", "Wc1", "Wc2")       # spectrally capped recurrent maps


def configure(cfg):
    """Set the module globals from a recipe dict (sudoku.train.RECIPE or a checkpoint's
    cfg): grid size N, channel width C and hidden ratios (D_T = round(r_t * P),
    D_C = round(r_c * C)), the activation act_spec, the LN variance floor norm_eps, and
    the fixed row/column/box one-hots. The block layout keys depth / inject must be
    1 / "all" (one mixer block, input injected at its main node). Call before
    init_params / embed."""
    if cfg.get("depth", 1) != 1 or cfg.get("inject", "all") != "all":
        raise ValueError(f"sudoku.model supports depth=1, inject='all' only, got "
                         f"depth={cfg.get('depth')!r}, inject={cfg.get('inject')!r}")
    global N, P, D_T, D_C, N_CLASSES, _POS_OH, F_IN, ACT, NORM_EPS
    N = cfg["N"]; P = N * N; N_CLASSES = N
    D_T = max(1, int(round(cfg["r_t"] * P))); D_C = max(1, int(round(cfg["r_c"] * cfg["C"])))
    row_id, col_id, box_id = _data.positional_ids(N)
    oh = lambda a: np.eye(N, dtype=np.float32)[a]
    _POS_OH = jnp.asarray(np.concatenate([oh(row_id), oh(col_id), oh(box_id)], axis=1))
    F_IN = (N + 1) + 3 * N
    ACT = cfg["act_spec"]
    NORM_EPS = float(cfg["norm_eps"])


def _norm(x):
    """In-loop channel norm: non-affine LayerNorm with variance floor NORM_EPS."""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + NORM_EPS)


def _ln_affine(x, g, b, eps=1e-5):
    """Affine LayerNorm over the channel axis (readout only)."""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * g + b


# ----------------------------- params --------------------------------
def init_params(key, C=256, init_scale=1.0):
    """theta (stem + the 4 mixing matrices, 1/sqrt(fan_in) init * init_scale; the caller
    applies project_spectral) and psi (per-cell readout)."""
    ks = jax.random.split(key, 7)        # Ws, the 4 mixing matrices, V, one unused key
    lin = lambda k, m, n: jax.random.normal(k, (m, n)) * (init_scale / jnp.sqrt(m))
    theta = {
        # stem: [clue one-hot (N+1) | row/col/box one-hots (3N)] -> C (feedforward)
        "Ws": jax.random.normal(ks[0], (F_IN, C)) * (1.0 / jnp.sqrt(F_IN)),
        "bs": jnp.zeros((C,)),
        # token-mixing MLP (across the P cells): constraint propagation
        "Wt1": lin(ks[1], P, D_T), "bt1": jnp.zeros((D_T,)),
        "Wt2": lin(ks[2], D_T, P), "bt2": jnp.zeros((P,)),
        # channel-mixing MLP (across the C features): per-cell reasoning
        "Wc1": lin(ks[3], C, D_C), "bc1": jnp.zeros((D_C,)),
        "Wc2": lin(ks[4], D_C, C), "bc2": jnp.zeros((C,)),
    }
    psi = {
        "g_head": jnp.ones((C,)), "b_head": jnp.zeros((C,)),
        "V": jax.random.normal(ks[5], (C, N_CLASSES)) * (1.0 / jnp.sqrt(C)),
        "c": jnp.zeros((N_CLASSES,)),
    }
    return {"theta": theta, "psi": psi}


# ----------------------------- model maps --------------------------------
def embed(theta, puz):
    """Stem: puzzle (B, P) ints 0..N -> x_inj (B, P, C): clue one-hot (N+1) ++
    fixed positional one-hots (3N), then the linear stem. Feedforward theta."""
    B = puz.shape[0]
    digit_oh = jax.nn.one_hot(puz, N + 1, dtype=theta["Ws"].dtype)       # (B, P, N+1)
    pos = jnp.broadcast_to(_POS_OH.astype(theta["Ws"].dtype), (B, P, 3 * N))
    feat = jnp.concatenate([digit_oh, pos], axis=-1)                     # (B, P, F_IN)
    return feat @ theta["Ws"] + theta["bs"]                              # (B, P, C)


def ht_hat(theta, s):
    """Token-mix hidden prediction: act(LN(s)^T @ Wt1) -> (B, C, D_T)."""
    u = _norm(s).transpose(0, 2, 1)                       # (B, C, P)
    return apply_act(ACT, u @ theta["Wt1"] + theta["bt1"])


def t_hat(theta, h_t):
    """Token-mix output prediction: (h_t @ Wt2)^T -> (B, P, C)."""
    o = h_t @ theta["Wt2"] + theta["bt2"]                 # (B, C, P)
    return o.transpose(0, 2, 1)


def hc_hat(theta, t):
    """Channel-mix hidden prediction: act(LN(t) @ Wc1) -> (B, P, D_C)."""
    return apply_act(ACT, _norm(t) @ theta["Wc1"] + theta["bc1"])


def s_hat(theta, h_c, x_inj):
    """Main-node prediction: channel-mix output + clue injection -> (B, P, C)."""
    o = h_c @ theta["Wc2"] + theta["bc2"]                 # (B, P, C)
    return o + x_inj


def init_state(theta, x_inj):
    """Zero (h_t, t, h_c, s) state, shapes from x_inj / theta."""
    B, P_, C = x_inj.shape
    D_t, D_c = theta["Wt1"].shape[1], theta["Wc1"].shape[1]
    return (jnp.zeros((B, C, D_t), x_inj.dtype), jnp.zeros((B, P_, C), x_inj.dtype),
            jnp.zeros((B, P_, D_c), x_inj.dtype), jnp.zeros((B, P_, C), x_inj.dtype))


def block_sweep(theta, z, x_inj):
    """One sequential (Gauss-Seidel) sweep h_t, t, h_c, s, each from the just-updated
    predecessor (h_t from the current s); iterating it is the forward pass of the
    equilibrium model, with fixed point the joint equilibrium (zero energy)."""
    s = z[-1]                                             # loop-closing main node
    h_t = ht_hat(theta, s)
    t = t_hat(theta, h_t)
    h_c = hc_hat(theta, t)
    s = s_hat(theta, h_c, x_inj)
    return (h_t, t, h_c, s)


def predict(theta, z, x_inj):
    """Jacobi map: every node's prediction from the current state, in z's order (h_t
    from z[-1], t from h_t, h_c from t, s from h_c and x_inj); energy_internal(z) =
    1/2 ||z - predict(z)||^2 term by term. Used by core.ep.coupled_theta_grad."""
    h_t, t, h_c, s = z
    return (ht_hat(theta, s), t_hat(theta, h_t), hc_hat(theta, t), s_hat(theta, h_c, x_inj))


def ring(theta, x_inj):
    """The loop as per-node prediction maps in z's order: node i is predicted from node
    i-1 alone (node 0 from the loop-closing z[-1]), so predict(z) == (fns[i](z[i-1])).
    Used by core.ep.nudged_phase."""
    return (lambda v: ht_hat(theta, v), lambda v: t_hat(theta, v),
            lambda v: hc_hat(theta, v), lambda v: s_hat(theta, v, x_inj))


def energy_internal(theta, z, x_inj):
    """E(z) = 1/2 sum of the 4 squared prediction residuals (batch and features
    summed); each term couples one node to its predecessor."""
    h_t, t, h_c, s = z
    r1 = h_t - ht_hat(theta, s)
    r2 = t - t_hat(theta, h_t)
    r3 = h_c - hc_hat(theta, t)
    r4 = s - s_hat(theta, h_c, x_inj)
    return 0.5 * (jnp.sum(r1 * r1) + jnp.sum(r2 * r2) + jnp.sum(r3 * r3) + jnp.sum(r4 * r4))


def var_penalty_pred(theta, z, x_inj, target=0.25):
    """Hinge relu(target - var)^2 on the per-token variance of the predictions t_hat and
    s_hat (the inputs of the in-loop LNs), pushing it up towards `target` so that the
    LN gain 1/sqrt(var) stays bounded. It is written on the predictions rather than on
    the state, so it depends on theta at fixed z_free and has a nonzero gradient inside
    the EP update; the forward map is unchanged."""
    h_t, _, h_c, _ = z
    vt = t_hat(theta, h_t).var(-1)                 # (B, P)
    vs = s_hat(theta, h_c, x_inj).var(-1)          # (B, P)
    hs = jnp.mean(jax.nn.relu(target - vs) ** 2)
    ht = jnp.mean(jax.nn.relu(target - vt) ** 2)
    return 0.5 * (hs + ht)


# ----------------------------- readout / loss --------------------------------
def readout(psi, s):
    """Per-cell readout: affine-LN(s) @ V -> logits (B, P, N_CLASSES). No pooling."""
    return _ln_affine(s, psi["g_head"], psi["b_head"]) @ psi["V"] + psi["c"]


def ce_loss(psi, z, y):
    """Mean per-cell cross-entropy of the readout of the main node against targets y
    (B, P) in 0..N-1: the supervised cost C(z)."""
    logp = jax.nn.log_softmax(readout(psi, z[-1]), axis=-1)
    nll = -jnp.take_along_axis(logp, y[..., None], axis=-1)[..., 0]     # (B, P)
    return jnp.mean(nll)


def cost_sum(psi, z, y):
    """Nudge cost: sum over the batch of the per-board mean cross-entropy. The batch sum
    keeps the latent dynamics of the samples independent; the mean over cells makes the
    per-cell nudge (hence beta) independent of the grid size."""
    logp = jax.nn.log_softmax(readout(psi, z[-1]), axis=-1)
    nll = -jnp.take_along_axis(logp, y[..., None], axis=-1)[..., 0]     # (B, P)
    return jnp.sum(jnp.mean(nll, axis=1))


def accuracy(psi, z, y):
    """(cell_acc, board_solve): per-cell top-1 accuracy and the fraction of boards
    solved completely (every cell correct)."""
    pred = jnp.argmax(readout(psi, z[-1]), axis=-1)       # (B, P)
    correct = (pred == y)
    return jnp.mean(correct), jnp.mean(jnp.all(correct, axis=1))


# --------------------------- spectral control ----------------------------
def _mat_sigma(W, key, n_iter=8):
    """Top singular value of a matrix W (power iteration)."""
    v = jax.random.normal(key, (W.shape[1],))
    v = v / (jnp.linalg.norm(v) + 1e-12)
    for _ in range(n_iter):
        u = W @ v; u = u / (jnp.linalg.norm(u) + 1e-12)
        v = W.T @ u; v = v / (jnp.linalg.norm(v) + 1e-12)
    return jnp.linalg.norm(W @ v)


def project_spectral(params, cap=1.0, n_iter=8):
    """Scale every mixing matrix (Wt1, Wt2, Wc1, Wc2) so that its spectral norm is
    <= cap, the contractivity knob (these are the recurrent maps of the loop Jacobian;
    the stem is a constant injection and is not capped). Applied after every weight
    update."""
    th = params["theta"]
    ks = jax.random.split(jax.random.PRNGKey(0), len(MIX_MATS))
    for name, k in zip(MIX_MATS, ks):
        s = _mat_sigma(th[name], k, n_iter)
        th = {**th, name: th[name] * jnp.minimum(1.0, cap / (s + 1e-8))}
    params["theta"] = th
    return params


def s_map(theta, x_inj):
    """The composed s -> s map of one sweep, s |-> s_hat(hc_hat(t_hat(ht_hat(s)))); its
    Jacobian governs the stability of the fixed point (core.diagnostics.spectral_radius)."""
    return lambda s: s_hat(theta, hc_hat(theta, t_hat(theta, ht_hat(theta, s))), x_inj)
