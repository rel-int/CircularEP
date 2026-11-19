"""Maps of the circular convolutional model for CIFAR and ImageNet32: a convolutional
encoder (stem) and a single conv block with two value nodes closed into a loop.
vision/train.py trains it, with a value node after every encoder layer as well; the
recipes are vision/recipes.py and vision/imagenet32/recipe.py.

State           z = (u, s)         u: inner value node (Ci channels), s: main value node (C)
Predictions     u_hat(s)   = act( conv1(s) )
                s_hat(u,x) = GN( act( conv2(u) + x_inj ) ),     x_inj = embed(x)
Energy          E(u,s) = 1/2||u - u_hat(s)||^2 + 1/2||s - s_hat(u,x)||^2
Readout         head((u+s)/2), head(h) = GN_affine(h) -> global average pool -> linear
                (needs Ci == C; the EP nudge reaches both value nodes)

The fixed point u = u_hat(s), s = s_hat(u, x) is the single-block conv DEQ
s = s_hat(u_hat(s), x). Design:
  * GroupNorm (non-affine) on the main node's prediction only: with a GroupNorm on every
    prediction the spectral cap on the convolutions would be inert (GN undoes their
    scaling); normalising only s keeps the cap a genuine contractivity knob.
  * No identity residual: the relaxation needs ||J|| < 1, which a residual breaks.
  * A multi-layer downsampling encoder maps x -> x_inj (embed); in vision/train.py every
    encoder layer has its own value node, initialised by this feedforward pass.

theta = {stem: [...], conv1, b1, conv2, b2}; psi = {g_head, bn_head, V, c}.

Module-level configuration, read at trace time (call before the first jit):
    configure(arch, C)       -> stem channel tuple for init_params; sets the strides,
                                kernel sizes and paddings of the encoder
    set_act(name)            in-loop activation (core.nn)
    set_in_loop_norm(mode)   "gn" (default) | "none"
    set_gn_var_floor(value)  None (default) | float: floor on the variance used by the
                             in-loop GroupNorm of s_hat
"""
import jax
import jax.numpy as jnp

from core.nn import apply_act

STEM_STRIDES = (1, 2, 2)
STEM_PAD = 1
PAD = 1
GROUPS = 8

# Per-encoder-layer kernel size / padding, parallel to STEM_STRIDES; set by configure().
STEM_KERNELS = (3, 3, 3)
STEM_PADS = (1, 1, 1)

IN_LOOP_NORM = "gn"
ACT = "relu"

# Floor on the within-group variance of the in-loop GroupNorm (the one inside s_hat;
# not the encoder's or the readout head's). None: normalise by the raw variance, whose
# gain 1/sqrt(var + eps) is unbounded as var -> 0. A float: normalise by
# max(var, GN_VAR_FLOOR). s_hat is shared by the free phase, the nudged phase, the
# energy and evaluation, so the floor applies to all of them. Set by set_gn_var_floor.
GN_VAR_FLOOR = None


def set_gn_var_floor(value):
    """Set the in-loop GroupNorm variance floor (see GN_VAR_FLOOR); None = no floor."""
    global GN_VAR_FLOOR
    GN_VAR_FLOOR = None if value is None else float(value)


ARCH = {
    "v1":      dict(strides=(1, 2, 2),       stem_ch=lambda C: (64, 128)),    # ->8x8
    # Doubled encoder width and 5x5 kernels on the first two encoder convolutions (pads
    # keep the spatial sizes 32/16/8 of "v1"); otherwise as "v1". ImageNet32 recipe.
    "v1_w2k5": dict(strides=(1, 2, 2),       stem_ch=lambda C: (128, 256),
                    kernels=(5, 5, 3), pads=(2, 2, 1)),
}


def set_act(name):
    """Set the in-loop activation by name (core.nn.apply_act)."""
    global ACT
    ACT = name


def set_in_loop_norm(mode):
    """Set the in-loop norm of s_hat: "gn" (non-affine GroupNorm) or "none"."""
    global IN_LOOP_NORM
    IN_LOOP_NORM = mode


def configure(arch, C):
    """Select an architecture of ARCH: set the encoder strides, kernel sizes and
    paddings (default 3x3, pad 1) and return the encoder channel tuple for width C
    (pass it to init_params)."""
    global STEM_STRIDES, STEM_KERNELS, STEM_PADS
    a = ARCH[arch]
    STEM_STRIDES = tuple(a["strides"])
    n_layers = len(STEM_STRIDES)
    STEM_KERNELS = tuple(a.get("kernels", (3,) * n_layers))
    STEM_PADS = tuple(a.get("pads", (STEM_PAD,) * n_layers))
    return a["stem_ch"](C)


# ----------------------------- primitives --------------------------------
def conv(x, w, b, stride, pad):
    """NHWC convolution with HWIO weights w, symmetric padding `pad`, optional bias b."""
    y = jax.lax.conv_general_dilated(
        x, w, (stride, stride), [(pad, pad), (pad, pad)],
        dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y if b is None else y + b


def groupnorm(h, gamma=None, beta=None, groups=GROUPS, eps=1e-5, var_floor=None):
    """GroupNorm over (H, W, channels of a group); gamma/beta None -> non-affine.
    var_floor: None normalises by the raw variance, a float by max(var, var_floor)
    (only s_hat passes it, see GN_VAR_FLOOR)."""
    B, H, W, Ch = h.shape
    hg = h.reshape(B, H, W, groups, Ch // groups)
    mu = hg.mean(axis=(1, 2, 4), keepdims=True)
    var = hg.var(axis=(1, 2, 4), keepdims=True)
    if var_floor is not None:
        var = jnp.maximum(var, var_floor)
    hg = ((hg - mu) / jnp.sqrt(var + eps)).reshape(B, H, W, Ch)
    return hg * gamma + beta if gamma is not None else hg


def _he(key, shape):
    fan_in = shape[0] * shape[1] * shape[2]
    return jax.random.normal(key, shape) * jnp.sqrt(2.0 / fan_in)


# ----------------------------- parameters -----------------------------
def init_params(key, in_ch=3, stem_ch=(64, 128), C=256, Ci=256, n_classes=10,
                init_scale=0.05):
    """stem: in_ch -> stem_ch[...] -> C; recurrent conv1/conv2; readout head.
    The readout head((u+s)/2) needs Ci == C (no extra parameters)."""
    assert Ci == C, f"readout head((u+s)/2) needs Ci == C, got Ci={Ci}, C={C}"
    assert C % GROUPS == 0 and Ci % GROUPS == 0
    chs = [in_ch] + list(stem_ch) + [C]
    assert len(chs) - 1 == len(STEM_STRIDES), (chs, STEM_STRIDES)
    keys = jax.random.split(key, len(chs) + 3)
    stem = []
    ks = STEM_KERNELS if len(STEM_KERNELS) == len(chs) - 1 else (3,) * (len(chs) - 1)
    for i in range(len(chs) - 1):
        k = ks[i]
        layer = {"w": _he(keys[i], (k, k, chs[i], chs[i + 1])),
                 "b": jnp.zeros((chs[i + 1],))}
        if i < len(chs) - 2:
            assert chs[i + 1] % GROUPS == 0
            layer["g"] = jnp.ones((chs[i + 1],))
            layer["bn"] = jnp.zeros((chs[i + 1],))
        stem.append(layer)
    theta = {
        "stem": stem,
        "conv1": jax.random.normal(keys[-3], (3, 3, C, Ci)) * init_scale,
        "b1": jnp.zeros((Ci,)),
        "conv2": jax.random.normal(keys[-2], (3, 3, Ci, C)) * init_scale,
        "b2": jnp.zeros((C,)),
    }
    psi = {
        "g_head": jnp.ones((C,)), "bn_head": jnp.zeros((C,)),
        "V": jax.random.normal(keys[-1], (C, n_classes)) * (1.0 / jnp.sqrt(C)),
        "c": jnp.zeros((n_classes,)),
    }
    return {"theta": theta, "psi": psi}


# ----------------------------- model maps --------------------------------
def stem_pads(n_layers):
    """Padding of each of the n_layers encoder convolutions."""
    return STEM_PADS if len(STEM_PADS) == n_layers else (STEM_PAD,) * n_layers


def stem_layer(layer, stride, pad, h):
    """One encoder layer: convolution, then relu + affine GroupNorm (all but the last
    layer, which is linear)."""
    h = conv(h, layer["w"], layer["b"], stride, pad)
    if "g" in layer:
        h = groupnorm(jax.nn.relu(h), layer["g"], layer["bn"])
    return h


def embed(theta, x):
    """Feedforward encoder: x (B,32,32,3) -> x_inj (B,8,8,C). vision/train.py uses its
    intermediates as the encoder value nodes of the free phase."""
    pads = stem_pads(len(theta["stem"]))
    h = x
    for i, layer in enumerate(theta["stem"]):
        h = stem_layer(layer, STEM_STRIDES[i], pads[i], h)
    return h


def u_hat(theta, s):
    """Inner-node prediction from the main node: act(conv1(s))."""
    return apply_act(ACT, conv(s, theta["conv1"], theta["b1"], 1, PAD))


def s_hat(theta, u, x_inj):
    """Main-node prediction from the inner node and the injection:
    GN(act(conv2(u) + x_inj)), with the variance floor GN_VAR_FLOOR."""
    h = apply_act(ACT, conv(u, theta["conv2"], theta["b2"], 1, PAD) + x_inj)
    return groupnorm(h, var_floor=GN_VAR_FLOOR) if IN_LOOP_NORM == "gn" else h


def init_state(theta, x_inj):
    """Zero (u, s) state, shapes from x_inj / theta."""
    B, H, W, C = x_inj.shape
    Ci = theta["conv1"].shape[3]
    return (jnp.zeros((B, H, W, Ci), x_inj.dtype), jnp.zeros_like(x_inj))


def block_sweep(theta, z, x_inj):
    """One Gauss-Seidel sweep u <- u_hat(s); s <- s_hat(u, x). Its fixed point is the
    joint equilibrium (zero energy)."""
    u, s = z
    u = u_hat(theta, s)
    s = s_hat(theta, u, x_inj)
    return (u, s)


def energy_internal(theta, z, x_inj):
    """E(u,s) = 1/2||u-u_hat(s)||^2 + 1/2||s-s_hat(u,x)||^2 (batch+feat sum)."""
    u, s = z
    ru = u - u_hat(theta, s)
    rs = s - s_hat(theta, u, x_inj)
    return 0.5 * jnp.sum(ru * ru) + 0.5 * jnp.sum(rs * rs)


def var_penalty(theta, z, x_inj, target=0.25):
    """Hinge relu(target - var)^2 on the within-group variance of s_hat's input to the
    in-loop GroupNorm, pushing it up towards `target` so that the GroupNorm gain
    1/sqrt(var) stays bounded. It changes training only, not the forward map, and
    depends on theta through conv2, so it has a nonzero gradient at fixed z."""
    u, s = z
    h_s = apply_act(ACT, conv(u, theta["conv2"], theta["b2"], 1, PAD) + x_inj)
    B, H, W, Ch = h_s.shape
    hg = h_s.reshape(B, H, W, GROUPS, Ch // GROUPS)
    var = hg.var(axis=(1, 2, 4))
    return jnp.mean(jax.nn.relu(target - var) ** 2)


# ----------------------------- readout / loss --------------------------------
def _head(h, g, bn, V, c):
    """affine GN -> global average pool -> linear. (B,H,W,Ch) -> (B, n_classes)."""
    return groupnorm(h, g, bn).mean(axis=(1, 2)) @ V + c


def readout_z(psi, z):
    """Logits from the full latent state z=(u,s): head((u+s)/2)."""
    u, s = z[0], z[-1]
    h = 0.5 * (u + s)
    return _head(h, psi["g_head"], psi["bn_head"], psi["V"], psi["c"])


def ce_loss(psi, z, y):
    """Mean cross-entropy of the readout against the labels y."""
    logp = jax.nn.log_softmax(readout_z(psi, z), axis=-1)
    nll = -jnp.take_along_axis(logp, y[:, None], axis=-1)[:, 0]
    return jnp.mean(nll)


def cost_sum(psi, z, y):
    """Batch sum (not mean) of the per-sample cross-entropy, so that the latent dynamics
    of the samples are independent. Its gradient w.r.t. z is the EP nudge; the readout
    reads (u+s)/2, so it is nonzero on both value nodes."""
    logp = jax.nn.log_softmax(readout_z(psi, z), axis=-1)
    return -jnp.sum(jnp.take_along_axis(logp, y[:, None], axis=-1))


def accuracy(psi, z, y):
    """Top-1 accuracy of the readout."""
    return jnp.mean(jnp.argmax(readout_z(psi, z), axis=-1) == y)


# --------------------------- spectral control ----------------------------
def _conv_sigma(w, in_shape, stride, pad, key, n_iter=10):
    """Operator spectral norm of the linear conv map, via power iteration
    with the conv and its transpose (VJP)."""
    f = lambda v: conv(v, w, None, stride, pad)
    v = jax.random.normal(key, (1,) + in_shape)
    v = v / (jnp.linalg.norm(v) + 1e-12)
    for _ in range(n_iter):
        av, vjp = jax.vjp(f, v)
        v = vjp(av)[0]
        v = v / (jnp.linalg.norm(v) + 1e-12)
    return jnp.linalg.norm(f(v))


def project_spectral(params, spatial=8, cap=0.5, n_iter=10):
    """Scale conv1/conv2 so that each linear conv map has spectral norm <= cap, the
    contractivity knob (only conv1/conv2 enter the loop Jacobian; the encoder is not
    projected). Applied after every weight update."""
    th = params["theta"]
    C, Ci = th["conv1"].shape[2], th["conv1"].shape[3]
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    s1 = _conv_sigma(th["conv1"], (spatial, spatial, C), 1, PAD, k1, n_iter)
    s2 = _conv_sigma(th["conv2"], (spatial, spatial, Ci), 1, PAD, k2, n_iter)
    params["theta"] = {**th,
                       "conv1": th["conv1"] * jnp.minimum(1.0, cap / (s1 + 1e-8)),
                       "conv2": th["conv2"] * jnp.minimum(1.0, cap / (s2 + 1e-8))}
    return params
