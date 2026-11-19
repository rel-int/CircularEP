"""CIFAR recipe of the circular convolutional model and its builders, used by vision.train.

RECIPE: the hyper-parameters of the CIFAR experiments (batch size 64, 180 epochs, coupled
nudged phase, in-loop GroupNorm variance floor equal to var_target). vision.train trains
CIFAR-10 and CIFAR-100 with it, plus the nudging settings of its dataset presets.
build_model sets vision.model's trace-time globals and builds the parameters; build_cfg
translates the recipe into the training configuration read by vision.train.run.
Importing this module sets the JAX matmul precision to TF32.
"""
import jax
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

from vision import data as D
from vision import model as M

N_CLASSES = {"cifar10": 10, "cifar100": 100}

RECIPE = dict(
    C=256, Ci=256, cap=5.018955419161522, act_spec="silu@1.340",
    beta=3.426809159220999, var_pen=0.4393662105236037,
    var_target=0.129565606138326, epochs=180, bs=64,
    lr=0.0007239597265780512, wd=0.00016354027641004516,
    lr_cycles=2, lr_floor_frac=0.0,
    T_free=40,              # sweeps of the plain evaluation
    m=9,                    # Anderson depth, training free phase and Anderson evaluation
    free_t=40,              # Anderson iterations, training free phase and Anderson evaluation
    c_T=20, c_m=0,          # coupled nudge: 20 plain Gauss-Seidel sweeps (presets override T)
    gn_var_floor=0.129565606138326,
)


def build_model(dataset, seed, r):
    """Configure vision.model's trace-time globals and build fresh, spectrally
    projected params (init_scale=0.1)."""
    stem_ch = M.configure("v1", r["C"])
    M.set_act(r["act_spec"])
    M.set_in_loop_norm("gn")
    M.set_gn_var_floor(r.get("gn_var_floor"))
    params = M.init_params(jax.random.PRNGKey(seed), stem_ch=stem_ch, C=r["C"],
                           Ci=r["Ci"], n_classes=N_CLASSES[dataset], init_scale=0.1)
    return M.project_spectral(params, cap=r["cap"])


def build_cfg(r, epochs, seed):
    """Translate the recipe into the training configuration of vision.train.run (see
    vision.train.DEFAULTS)."""
    return dict(
        epochs=epochs, bs=r["bs"], lr=r["lr"], wd=r["wd"],
        lr_cycles=r["lr_cycles"], lr_floor_frac=r["lr_floor_frac"], seed=seed,
        beta=r["beta"], mode="random", form="coupled",
        free=dict(solver="anderson", T=r["free_t"], m=r["m"], reduced=True),
        nudge=dict(T=r["c_T"], m=r["c_m"]),
        eval_solver="both",
        eval_free=dict(solver="sweep", T=r["T_free"]),
        eval_free_aa=dict(solver="anderson", reduced=True, m=r["m"], T=r["free_t"]),
        var_pen=r["var_pen"], var_target=r["var_target"], cap=r["cap"],
        report_every=1, aug_fn=D.augment,
        # model description, stored with the checkpoints (not read by the trainer)
        act_spec=r["act_spec"], arch="v1",
    )
