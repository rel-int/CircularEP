"""The outer optimiser: clip -> decoupled weight decay -> Adam on a warmup +
cosine-decay schedule, the per-step parameter update (per-leaf learning-rate
scaling of the ring's token- vs channel-mixing matrices, then the spectral
projection), and an EMA accumulator of the trained parameters for evaluation.
"""
import jax
import jax.numpy as jnp
import optax

from recursive import model as M


def build_schedule(cfg, total_steps):
    """Linear warmup from 0 to cfg["lr"] over `warmup` steps (cfg["warmup"], 0 =
    min(200, total_steps // 10), clamped to total_steps // 2), then cosine decay
    to cfg["lr_min"] reached at total_steps."""
    warmup = min(cfg["warmup"] or min(200, max(1, total_steps // 10)),
                 max(1, total_steps // 2))
    sched = optax.warmup_cosine_decay_schedule(
        0.0, cfg["lr"], warmup, total_steps, end_value=cfg["lr_min"])
    return sched, warmup


def build_optimizer(cfg, total_steps):
    """optax.chain(clip_by_global_norm(1.0), add_decayed_weights(cfg["wd"]),
    adam(schedule, b1=0.9)) over `total_steps` optimiser steps."""
    sched, _ = build_schedule(cfg, total_steps)
    return optax.chain(optax.clip_by_global_norm(1.0),
                       optax.add_decayed_weights(cfg["wd"]),
                       optax.adam(sched, b1=0.9))


def apply_update(opt, params, grads, opt_state, cfg):
    """optimiser step -> per-leaf lr scaling of the update (TOK_LEAVES scaled by
    cfg["tok_lr_scale"], CHAN_LEAVES by cfg["chan_lr_scale"]; 1.0 = no-op) ->
    spectral projection at cfg["cap"]. Returns (params, opt_state)."""
    updates, opt_state = opt.update(grads, opt_state, params)
    new = optax.apply_updates(params, updates)
    th = dict(new["theta"])
    for scale, leaves in ((cfg["tok_lr_scale"], M.TOK_LEAVES),
                         (cfg["chan_lr_scale"], M.CHAN_LEAVES)):
        if scale != 1.0:
            for k in leaves:
                th[k] = params["theta"][k] + scale * (th[k] - params["theta"][k])
    new = {**new, "theta": th}
    new = M.project_spectral(new, cap=cfg["cap"], n_iter=8)
    return new, opt_state


class EMA:
    """Exponential moving average of a parameter pytree, bias-corrected at read
    time (Adam-style: corrected = raw_average / (1 - decay**steps))."""

    def __init__(self, decay):
        self.decay = float(decay)
        self.params = None
        self.steps = 0

    def update(self, params):
        """Fold `params` into the average."""
        d = self.decay
        if self.params is None:
            self.params = jax.tree_util.tree_map(lambda q: (1.0 - d) * q, params)
        else:
            self.params = jax.tree_util.tree_map(
                lambda e, q: d * e + (1.0 - d) * q, self.params, params)
        self.steps += 1

    def corrected(self):
        """The bias-corrected average."""
        bias = 1.0 - self.decay ** self.steps
        return jax.tree_util.tree_map(lambda e: jnp.asarray(e) / bias, self.params)
