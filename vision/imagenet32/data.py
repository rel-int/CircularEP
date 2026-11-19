"""ImageNet 32x32 (downsampled ImageNet-1k, 1000 classes) loading. The cache
(train_x.npy/train_y.npy/val_x.npy/val_y.npy/stats.json) lives under
`data/imagenet32` in the repository root; override with the `root` argument
(vision.train --data_dir) or the IMAGENET32_ROOT environment variable.

Source: the ungated HuggingFace mirror `benjamin-paine/imagenet-1k-32x32`
(ILSVRC-2012 center-cropped to square and Lanczos-downsampled to 32x32;
5 parquet files, 0.67 GB; `image` = PNG bytes, `label` = 0..999, plus a 1001st
class "none" used only by the UNLABELED test split, which we ignore):

    train       1,281,167 images   data/train-0000{0,1,2}-of-00003.parquet
    validation     50,000 images   data/validation-00000-of-00001.parquet
    test          100,000 images   (unlabeled -- not used)

As in every ImageNet32 paper (Chrabaszcz et al. 2017), the 50k VALIDATION split
is the held-out test set.

`python -m vision.imagenet32.prepare` downloads the parquets once, decodes the
PNGs in parallel and caches them as raw uint8 .npy arrays (train 3.9 GB, val
154 MB) plus the per-channel train mean/std in stats.json. `get_imagenet32()`
then memory-maps the cache and returns UINT8 NHWC images + int32 labels
(+ mean/std); unlike vision/data.py the float conversion + normalisation is
done per batch on the device (`normalize`), because 1.28M images x 3072 x
float32 = 15.7 GB would not fit comfortably alongside the model, whereas 3.9 GB
of uint8 does. The TRAIN array is therefore kept on the host (see
vision/imagenet32/recipe.py's `make_batch_fn`).
"""
import json
import os

import numpy as np
import jax.numpy as jnp

DATA_ROOT = os.environ.get(
    "IMAGENET32_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "imagenet32"))

HF_REPO = "benjamin-paine/imagenet-1k-32x32"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/data/{{name}}"
PARQUETS = {
    "train": ["train-00000-of-00003.parquet",
              "train-00001-of-00003.parquet",
              "train-00002-of-00003.parquet"],
    "validation": ["validation-00000-of-00001.parquet"],
}
N_CLASSES = 1000
N_TRAIN, N_VAL = 1_281_167, 50_000

# standard ImageNet per-channel stats (RGB, [0,1] scale); prepare.py
# overwrites them with the measured 32x32 train stats in stats.json
_MEAN_DEFAULT = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD_DEFAULT = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CACHE = {
    "train_x": "train_x.npy", "train_y": "train_y.npy",
    "val_x": "val_x.npy", "val_y": "val_y.npy",
    "stats": "stats.json",
}


def cache_paths(root=None):
    root = os.path.expanduser(root or DATA_ROOT)
    return {k: os.path.join(root, v) for k, v in CACHE.items()}


def cache_ready(root=None):
    return all(os.path.exists(p) for p in cache_paths(root).values())


def load_stats(root=None):
    p = cache_paths(root)["stats"]
    if os.path.exists(p):
        s = json.load(open(p))
        return (np.array(s["mean"], dtype=np.float32),
                np.array(s["std"], dtype=np.float32))
    return _MEAN_DEFAULT, _STD_DEFAULT


def get_imagenet32(root=None, device=True):
    """Return (x_train u8, y_train i32, x_val u8, y_val i32, mean, std).
    Images are (N, 32, 32, 3) uint8 NHWC; with device=True they are moved to
    the default JAX device (3.9 GB + 154 MB), otherwise left as np memmaps.
    vision.train calls this with device=False and keeps x_train on the host
    (see module docstring)."""
    P = cache_paths(root)
    if not cache_ready(root):
        root = os.path.dirname(P["train_x"])
        raise FileNotFoundError(
            f"ImageNet32 cache missing under {root}; run "
            f"`python -m vision.imagenet32.prepare --data_dir {root}` first")
    xtr = np.load(P["train_x"], mmap_mode="r")
    ytr = np.load(P["train_y"])
    xva = np.load(P["val_x"], mmap_mode="r")
    yva = np.load(P["val_y"])
    assert xtr.dtype == np.uint8 and xtr.shape[1:] == (32, 32, 3), xtr.shape
    assert ytr.min() >= 0 and ytr.max() < N_CLASSES, (ytr.min(), ytr.max())
    mean, std = load_stats(root)
    if device:
        xtr, xva = jnp.asarray(np.ascontiguousarray(xtr)), jnp.asarray(
            np.ascontiguousarray(xva))
        ytr, yva = jnp.asarray(ytr), jnp.asarray(yva)
    return xtr, ytr, xva, yva, mean, std


def normalize(x_u8, mean, std):
    """uint8 NHWC -> normalised float32 (the same transform vision/data.py
    bakes into its arrays), as a jnp op so it can live inside the jitted
    batch path."""
    return (x_u8.astype(jnp.float32) / 255.0 - jnp.asarray(mean)) / jnp.asarray(std)
