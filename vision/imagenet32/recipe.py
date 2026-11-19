"""ImageNet32 recipe of the circular convolutional model and its builders, used by
vision.train.

RECIPE: architecture "v1_w2k5" (encoder width 128/256 and 5x5 kernels on the first two
encoder convolutions, see vision/model.py), width 512, batch size 128, 50 epochs, in-loop
GroupNorm variance floor 0.09184 (= var_target). build_model / build_cfg mirror
vision/recipes.py. The training set stays in host memory and is batched by the
`batch_fn` hook of the training configuration (make_batch_fn, wrapped in
PicklableBatchFn). The data cache is built by `python -m vision.imagenet32.prepare`.
Importing this module sets the JAX matmul precision to TF32.
"""
import jax
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
import jax.numpy as jnp
import numpy as np

from vision import model as M
from vision.imagenet32 import data as D

N_CLASSES = D.N_CLASSES  # 1000

# Keys as in vision/recipes.py's RECIPE.
RECIPE = dict(
    arch="v1_w2k5",
    C=512, Ci=512,
    cap=4.977310725929158,
    act_spec="silu@1.627",
    beta=3.173292679845857,
    var_pen=0.03516111504803029,
    var_target=0.0918400469561869,
    gn_var_floor=0.09184,
    epochs=50, bs=128,
    lr=0.00031075538782674643, wd=2.6e-05,
    lr_cycles=1, lr_floor_frac=0.0,
    T_free=40,              # sweeps of the plain evaluation (the selection metric)
    m=9,                    # Anderson depth of the training free phase (main value node)
    free_t=25,              # Anderson iterations of the training free phase
    c_T=10, c_m=0,          # coupled nudge: 10 plain Gauss-Seidel sweeps
)


class PicklableBatchFn:
    """The training loop pickles its configuration (including cfg["batch_fn"]) into
    <tag>_best.pkl, and a closure is not picklable. This wrapper delegates calls and
    pickles as the string "batch_fn" (it never serialises the training array)."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, key, idx):
        return self.fn(key, idx)

    def __reduce__(self):
        return (str, ("batch_fn",))


def build_model(seed, r):
    """Configure vision.model's trace-time globals and build fresh, spectrally
    projected params (init_scale=0.1, n_classes=1000)."""
    stem_ch = M.configure(r.get("arch", "v1"), r["C"])
    M.set_act(r["act_spec"])
    M.set_in_loop_norm("gn")
    M.set_gn_var_floor(r.get("gn_var_floor"))
    params = M.init_params(jax.random.PRNGKey(seed), stem_ch=stem_ch, C=r["C"],
                           Ci=r["Ci"], n_classes=N_CLASSES, init_scale=0.1)
    return M.project_spectral(params, cap=r["cap"])


def build_cfg(r, epochs, seed):
    """Translate the recipe into the training configuration of vision.train.run (see
    vision.train.DEFAULTS); same keys as vision/recipes.py's build_cfg. The caller adds
    cfg["batch_fn"] (see make_batch_fn)."""
    return dict(
        epochs=epochs, bs=r["bs"], lr=r["lr"], wd=r["wd"],
        lr_cycles=r["lr_cycles"], lr_floor_frac=r["lr_floor_frac"], seed=seed,
        beta=r["beta"], mode="random", form="coupled",
        free=dict(solver="anderson", T=r["free_t"], m=r["m"], reduced=True),
        nudge=dict(T=r["c_T"], m=r["c_m"]),
        eval_solver="plain",
        eval_free=dict(solver="sweep", T=r["T_free"]),
        var_pen=r["var_pen"], var_target=r["var_target"], cap=r["cap"],
        report_every=1,
        # model description, stored with the checkpoints (not read by the trainer)
        act_spec=r["act_spec"], arch=r["arch"],
    )


# The ImageNet32 training set is 1.28M x 32x32x3 uint8 (3.9 GB). It stays in host memory:
# each batch is gathered with numpy outside jit, and only the small uint8 batch is moved
# to the device, where normalisation and augmentation run inside one jitted function.
def make_batch_fn(XTR, YTR, MEAN, STD, aug_fn):
    """The `cfg["batch_fn"](key, idx) -> (x, y)` hook for the host-resident uint8
    training array: `idx` (a slice of the per-epoch permutation) is sorted for a
    contiguous host gather XTR[idx], then the uint8 batch is moved to the device and
    normalised + augmented in one jitted function."""
    norm_aug = jax.jit(lambda key, x_u8: aug_fn(key, D.normalize(x_u8, MEAN, STD)))

    def batch_fn(key, idx):
        idx_np = np.sort(np.asarray(idx))   # sorted: contiguous host gather
        x_u8 = jnp.asarray(XTR[idx_np])      # host -> device: bs * 3072 bytes
        y = jnp.asarray(YTR[idx_np])
        return norm_aug(key, x_u8), y

    return batch_fn
