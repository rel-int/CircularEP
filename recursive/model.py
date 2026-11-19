"""The recursive ring: a single attention-free MLP-Mixer DEQ block (token-mixing
MLP replaces self-attention; channel-mixing MLP is the FFN) run to a fixed point,
with the carried ANSWER STATE y injected at the main node alongside the clue stem.

State z = (h_t, t, h_c, s), one value node per linear sub-layer, chained into a loop
that closes at s (s both feeds and is fed by the chain):

    h_t = act(LN(s)^T Wt1 + bt1),         t   = (h_t Wt2 + bt2)^T
    h_c = act(LN(t) Wc1 + bc1),           s   = h_c Wc2 + bc2 + x_inj

    x_inj = embed_base(puz) + LN(y) Wy

`y` is the free-equilibrium value of s carried from the previous minimisation
(clamped during a minimisation, exactly like the clue injection x_inj): with
y = 0 (LN(0) = 0) the injection vanishes, so the first stage of every datapoint
is exactly the plain ring.

P = N*N cells are the tokens; the channel dim C carries the per-cell digit
belief. Token-mixing propagates constraints across cells (a linear map, hence
spectrally cappable -- the contractivity knob); channel-mixing does per-cell
digit reasoning. The stem (clue one-hot + fixed row/col/box one-hots, forced
by the N x N Sudoku geometry) is feedforward theta, recomputed inside every
EP theta-grad, so the stem receives its own gradient.

`configure(cfg)` must be called once before `init_params` / `embed` / ...: it
sets the grid size, channel width ratios, activation and LN variance floor
from the recipe's own cfg dict.
"""
import jax
import jax.numpy as jnp
import numpy as np

from core.nn import apply_act
from sudoku.data import positional_ids

# -------- geometry (set by configure) --------
N = 9
P = 81
D_T = 162           # token-mix hidden width
D_C = 240           # channel-mix hidden width
N_CLASSES = 9
_POS_OH = None       # fixed positional one-hots (P, 3N), constant w.r.t. theta
F_IN = 0             # stem input feature dim = (N + 1) + 3N

# in-loop pre-norm variance floor (the LN gain is 1/sqrt(var + NORM_EPS); this is
# the term that lets rho(J) blow past the spectral cap if left too small)
NORM_EPS = 1e-5

# in-loop activation spec, "base" or "base@t" (temperature-scaled, see core.nn.apply_act)
ACT = "gelu"

# per-leaf lr groups (optimizers.apply_update) and the spectrally capped maps
TOK_LEAVES = ("Wt1", "bt1", "Wt2", "bt2")
CHAN_LEAVES = ("Wc1", "bc1", "Wc2", "bc2")
MIX_MATS = ("Wt1", "Wt2", "Wc1", "Wc2")


def configure(cfg):
    """Set the grid size N, the channel width ratios and the in-loop activation /
    LN variance floor from cfg (keys N, C, r_t, r_c, act_spec, norm_eps); build the
    fixed row/col/box positional one-hots. Call before init_params / embed / ..."""
    global N, P, D_T, D_C, N_CLASSES, _POS_OH, F_IN, ACT, NORM_EPS
    N = int(cfg["N"]); P = N * N; N_CLASSES = N
    D_T = max(1, int(round(cfg["r_t"] * P)))
    D_C = max(1, int(round(cfg["r_c"] * cfg["C"])))
    row_id, col_id, box_id = positional_ids(N)
    oh = lambda a: np.eye(N, dtype=np.float32)[a]
    _POS_OH = jnp.asarray(np.concatenate([oh(row_id), oh(col_id), oh(box_id)], axis=1))
    F_IN = (N + 1) + 3 * N
    ACT = cfg["act_spec"]
    NORM_EPS = float(cfg["norm_eps"])


def _norm(x, eps=None):
    """In-loop channel norm: non-affine LayerNorm, variance floor NORM_EPS (bounds
    the LN gain by 1/sqrt(NORM_EPS))."""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + (NORM_EPS if eps is None else eps))


def _ln_affine(x, g, b, eps=1e-5):
    """Affine LayerNorm over the channel axis (readout only)."""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * g + b


# ----------------------------- params --------------------------------
def init_params(key, cfg):
    """theta (stem + the 4 ring matrices + the answer-injection Wy, all
    1/sqrt(fan_in) init * cfg["gain"], Wy at zero, then capped by
    project_spectral) and psi (per-cell readout)."""
    C, init_scale = cfg["C"], cfg["gain"]
    ks = jax.random.split(key, 7)        # Ws, the 4 ring matrices, V, one unused key
    lin = lambda k, m, n: jax.random.normal(k, (m, n)) * (init_scale / jnp.sqrt(m))
    theta = {
        "Ws": jax.random.normal(ks[0], (F_IN, C)) * (1.0 / jnp.sqrt(F_IN)),
        "bs": jnp.zeros((C,)),
        "Wt1": lin(ks[1], P, D_T), "bt1": jnp.zeros((D_T,)),
        "Wt2": lin(ks[2], D_T, P), "bt2": jnp.zeros((P,)),
        "Wc1": lin(ks[3], C, D_C), "bc1": jnp.zeros((D_C,)),
        "Wc2": lin(ks[4], D_C, C), "bc2": jnp.zeros((C,)),
        "Wy": jnp.zeros((C, C)),
    }
    psi = {
        "g_head": jnp.ones((C,)), "b_head": jnp.zeros((C,)),
        "V": jax.random.normal(ks[5], (C, N_CLASSES)) * (1.0 / jnp.sqrt(C)),
        "c": jnp.zeros((N_CLASSES,)),
    }
    return {"theta": theta, "psi": psi}


def count_params(params):
    """Total number of scalars in theta + psi."""
    return int(sum(a.size for a in jax.tree_util.tree_leaves(params)))


# ----------------------------- model maps --------------------------------
def embed(theta, x, y):
    """Stem + answer injection: x_inj = embed_base(puz) + LN(y) @ Wy. x: puzzle
    (B, P) ints 0..N (0 = blank); y: the carried answer state (B, P, C)."""
    B = x.shape[0]
    digit_oh = jax.nn.one_hot(x, N + 1, dtype=theta["Ws"].dtype)        # (B, P, N+1)
    pos = jnp.broadcast_to(_POS_OH.astype(theta["Ws"].dtype), (B, P, 3 * N))
    feat = jnp.concatenate([digit_oh, pos], axis=-1)                    # (B, P, F_IN)
    x_inj = feat @ theta["Ws"] + theta["bs"]
    return x_inj + _norm(y) @ theta["Wy"]


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
    """Main-node prediction (channel-mix output + clue/answer injection)."""
    return h_c @ theta["Wc2"] + theta["bc2"] + x_inj


def init_state(theta, x_inj):
    """Zero (h_t, t, h_c, s) state, shapes from x_inj / theta."""
    B, P_, C = x_inj.shape
    D_t, D_c = theta["Wt1"].shape[1], theta["Wc1"].shape[1]
    return (jnp.zeros((B, C, D_t), x_inj.dtype), jnp.zeros((B, P_, C), x_inj.dtype),
            jnp.zeros((B, P_, D_c), x_inj.dtype), jnp.zeros((B, P_, C), x_inj.dtype))


def block_sweep(theta, z, x_inj):
    """One Gauss-Seidel sweep: h_t from the current s, t from the new h_t, h_c from
    the new t, s from the new h_c (+x_inj); the loop closes because s both feeds
    and is fed by this chain. Iterating it = the DEQ forward pass."""
    _, _, _, s = z
    h_t = ht_hat(theta, s)
    t = t_hat(theta, h_t)
    h_c = hc_hat(theta, t)
    s = s_hat(theta, h_c, x_inj)
    return (h_t, t, h_c, s)


def energy_internal(theta, z, x_inj):
    """E(z) = 1/2 sum of the 4 squared prediction residuals (batch + feature
    summed): each value node against its own prediction from its predecessor."""
    h_t, t, h_c, s = z
    r1 = h_t - ht_hat(theta, s)
    r2 = t - t_hat(theta, h_t)
    r3 = h_c - hc_hat(theta, t)
    r4 = s - s_hat(theta, h_c, x_inj)
    e = jnp.sum(r1 * r1) + jnp.sum(r2 * r2) + jnp.sum(r3 * r3) + jnp.sum(r4 * r4)
    return 0.5 * e


def var_penalty_pred(theta, z, x_inj, target):
    """EP-effective lambda_max hinge: push the per-token variance the in-loop LNs
    see (on the t and s PREDICTIONS, theta-dependent at fixed z) up toward
    `target`, bounding the in-loop LN gain 1/sqrt(var) without touching the
    forward map. Nonzero under EP (z_free held fixed); the state-variable version
    of this penalty has an identically zero EP theta-grad and is not used here."""
    h_t, _, h_c, _ = z
    vt = t_hat(theta, h_t).var(-1)                 # (B, P)
    vs = s_hat(theta, h_c, x_inj).var(-1)          # (B, P)
    hs = jnp.mean(jax.nn.relu(target - vs) ** 2)
    ht = jnp.mean(jax.nn.relu(target - vt) ** 2)
    return 0.5 * (hs + ht)


# ----------------------------- readout / loss --------------------------------
def readout(psi, s):
    """Per-cell readout: affine-LN(s) @ V -> logits (B, P, N_CLASSES)."""
    return _ln_affine(s, psi["g_head"], psi["b_head"]) @ psi["V"] + psi["c"]


def ce_loss(psi, z, y):
    """Mean per-cell CE of the readout on the main value node vs targets y (B, P) in
    0..N-1. The supervised cost C(z)."""
    logp = jax.nn.log_softmax(readout(psi, z[-1]), axis=-1)
    nll = -jnp.take_along_axis(logp, y[..., None], axis=-1)[..., 0]
    return jnp.mean(nll)


def cost_sum(psi, z, y):
    """Nudge cost: SUM over the batch of the MEAN per-cell CE (batch-sum keeps the
    per-sample latent dynamics independent inside a fused multi-sample
    relaxation; the per-cell average makes beta grid-size independent)."""
    logp = jax.nn.log_softmax(readout(psi, z[-1]), axis=-1)
    nll = -jnp.take_along_axis(logp, y[..., None], axis=-1)[..., 0]
    return jnp.sum(jnp.mean(nll, axis=1))


def accuracy(psi, z, y):
    """(cell_acc, board_solve): per-cell top-1 accuracy and the fraction of boards
    solved completely (every cell correct)."""
    pred = jnp.argmax(readout(psi, z[-1]), axis=-1)
    correct = (pred == y)
    return jnp.mean(correct), jnp.mean(jnp.all(correct, axis=1))


def answer_update(y, s_free, y_rate=1.0):
    """End-of-minimisation answer update from the FREE equilibrium of the readout
    value node: y <- (1 - y_rate) y + y_rate s_free (y_rate = 1 overwrites)."""
    if y_rate == 1.0:
        return s_free
    return (1.0 - y_rate) * y + y_rate * s_free


def rel_change(y_new, y_old):
    """Batch-mean relative change ||y_new - y_old|| / ||y_new|| -- the outer
    fixed-point residual of the answer chain."""
    B = y_new.shape[0]
    d = jnp.linalg.norm((y_new - y_old).reshape(B, -1), axis=1)
    n = jnp.linalg.norm(y_new.reshape(B, -1), axis=1)
    return jnp.mean(d / (n + 1e-12))


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
    """Scale every ring matrix (Wt1, Wt2, Wc1, Wc2) so its spectral norm <= cap --
    the single contractivity knob (Wy and the stem are outside the loop
    Jacobian, never capped)."""
    th = params["theta"]
    ks = jax.random.split(jax.random.PRNGKey(0), len(MIX_MATS))
    for name, k in zip(MIX_MATS, ks):
        s = _mat_sigma(th[name], k, n_iter)
        th = {**th, name: th[name] * jnp.minimum(1.0, cap / (s + 1e-8))}
    return {**params, "theta": th}
