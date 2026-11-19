"""CIFAR-10 / CIFAR-100 loading and on-device augmentation.

The whole dataset sits in device memory as NHWC float32 arrays with the per-channel
normalisation applied, so batching and augmentation run in JAX with no data loader in the
training loop. Both datasets are 32x32x3, 50k train / 10k test images.

Data directory: data/cifar under the repository root (or the `root` argument). CIFAR-10
is downloaded by torchvision if absent; CIFAR-100 from the HuggingFace parquet mirror of
the dataset if absent.
"""
import io
import os
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import jax
import jax.numpy as jnp
import pyarrow.parquet as pq
from PIL import Image

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "cifar")

_MEAN10 = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_STD10 = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
_MEAN100 = np.array([0.5071, 0.4865, 0.4409], dtype=np.float32)
_STD100 = np.array([0.2673, 0.2564, 0.2762], dtype=np.float32)


def _to_arrays(ds, mean, std):
    x = ds.data.astype(np.float32) / 255.0
    x = (x - mean) / std
    y = np.array(ds.targets).astype(np.int32)
    return x, y


def get_cifar10(root: str = DEFAULT_ROOT):
    """Return (x_train, y_train, x_test, y_test) as jnp NHWC arrays (10 classes)."""
    import torchvision
    root = os.path.expanduser(root)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        train = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        test = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    xtr, ytr = _to_arrays(train, _MEAN10, _STD10)
    xte, yte = _to_arrays(test, _MEAN10, _STD10)
    return jnp.asarray(xtr), jnp.asarray(ytr), jnp.asarray(xte), jnp.asarray(yte)


# CIFAR-100 is read from the uoft-cs/cifar100 parquet mirror on HuggingFace.
_HF_CIFAR100_URL = ("https://huggingface.co/datasets/uoft-cs/cifar100/resolve/"
                    "main/cifar100/{split}-00000-of-00001.parquet")


def _download(url, dest):
    """Download `url` to `dest` unless it exists (written to dest.part, then renamed)."""
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    return dest


def _decode_parquet_split(path):
    table = pq.read_table(path, columns=["img", "fine_label"])
    x = np.stack([np.array(Image.open(io.BytesIO(im["bytes"])))
                 for im in table.column("img").to_pylist()])
    y = np.array(table.column("fine_label").to_pylist(), dtype=np.int32)
    return x, y


def get_cifar100(root: str = DEFAULT_ROOT):
    """Return (x_train, y_train, x_test, y_test) as jnp NHWC arrays (100
    fine labels), from the HF parquet mirror."""
    root = os.path.expanduser(root)
    xtr, ytr = _decode_parquet_split(_download(
        _HF_CIFAR100_URL.format(split="train"),
        os.path.join(root, "cifar100_train.parquet")))
    xte, yte = _decode_parquet_split(_download(
        _HF_CIFAR100_URL.format(split="test"),
        os.path.join(root, "cifar100_test.parquet")))
    xtr = (xtr.astype(np.float32) / 255.0 - _MEAN100) / _STD100
    xte = (xte.astype(np.float32) / 255.0 - _MEAN100) / _STD100
    return jnp.asarray(xtr), jnp.asarray(ytr), jnp.asarray(xte), jnp.asarray(yte)


def augment(key, x, pad=4):
    """Random crop (reflect-pad `pad`, crop back) + random horizontal flip,
    vectorised over the batch. x: (B, 32, 32, 3)."""
    B, H, W, C = x.shape
    kc, kf = jax.random.split(key)
    xp = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
    offs = jax.random.randint(kc, (B, 2), 0, 2 * pad + 1)

    def crop_one(img, off):
        return jax.lax.dynamic_slice(img, (off[0], off[1], 0), (H, W, C))

    x = jax.vmap(crop_one)(xp, offs)
    flip = jax.random.bernoulli(kf, 0.5, (B, 1, 1, 1))
    return jnp.where(flip, x[:, :, ::-1, :], x)
