"""Shared in-loop activations, selected by NAME so each model keeps a global ACT.

A SMOOTH activation (gelu/silu/softplus/mish) removes the ReLU kinks that bias the
centered-EP finite difference on trained models. The optional "base@t" temperature
applies (1/t)*base(t*x): it keeps the +/-inf asymptotes (slopes 1/0) while tuning
kink sharpness (t->inf == relu, t->0 == linear = maximally smooth = most EP-faithful).
"""
import jax
import jax.numpy as jnp

_BASE = {
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "silu": jax.nn.silu,
    "softplus": jax.nn.softplus,
    "mish": lambda z: z * jnp.tanh(jax.nn.softplus(z)),
    "tanh": jnp.tanh,
}


def apply_act(name, x):
    """Apply the activation `name` ("relu"/"gelu"/"silu"/"softplus"/"mish"/"tanh",
    optionally temperature-scaled as "base@t")."""
    if "@" in name:
        base, t = name.split("@")
        if base == "relu":                       # relu is scale-invariant
            return jax.nn.relu(x)
        t = float(t)
        return _BASE[base](t * x) / t
    return _BASE[name](x)
