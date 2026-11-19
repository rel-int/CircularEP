"""Training utilities shared by the trainers: the warm-restart cosine learning-rate
schedule and chunked accuracy evaluation of an EP model (vision/train.py)."""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import ep as E


# ------------------------------- schedule -------------------------------
def make_schedule(lr, lr_cycles, total_steps, floor_frac):
    """Linear warmup (at most 200 steps) + cosine decay from `lr` over `total_steps` =
    epochs * steps_per_epoch, restarted `lr_cycles` times, with a flat floor at
    floor_frac * lr (floor_frac 0 = no floor)."""
    if lr_cycles > 1:
        cyc_steps = max(2, total_steps // lr_cycles)
        cyc_warmup = min(200, max(1, cyc_steps // 10), cyc_steps - 1)
        _cyc = optax.warmup_cosine_decay_schedule(0.0, lr, cyc_warmup, cyc_steps)
        _s0 = lambda k: _cyc(k % cyc_steps)
    else:
        warm = min(200, max(1, total_steps // 2))
        _s0 = optax.warmup_cosine_decay_schedule(0.0, lr, warm, total_steps)
    if floor_frac <= 0.0:
        return _s0
    floor = floor_frac * lr
    return lambda k: jnp.maximum(_s0(k), floor)


# --------------------------------- eval ---------------------------------
def make_eval_fn(model, free_cfg):
    """Jitted accuracy (core.ep.ep_eval) after a free phase with `free_cfg`."""
    return jax.jit(lambda p, x, y: E.ep_eval(model, p, x, y, free_cfg))


def evaluate(eval_fn, params, X, Y, chunk=1000):
    """Accuracy of `eval_fn` (see make_eval_fn) over (X, Y), in chunks of `chunk`
    samples: the mean of the chunk accuracies weighted by chunk size. When every chunk
    has the same size (e.g. 10k / 50k test images, chunk 1000) this is the plain mean,
    computed as such."""
    N = X.shape[0]
    accs, sizes = [], []
    for j in range(0, N, chunk):
        accs.append(float(eval_fn(params, X[j:j + chunk], Y[j:j + chunk])))
        sizes.append(min(chunk, N - j))
    if len(set(sizes)) == 1:
        return float(np.mean(accs))
    return float(np.dot(accs, sizes) / N)
