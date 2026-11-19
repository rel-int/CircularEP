"""Download and cache ImageNet32: fetch the parquet files from HuggingFace, decode the PNGs
in parallel and write raw uint8 arrays + train-set channel statistics under --data_dir
(default data/imagenet32 in the repository root, or $IMAGENET32_ROOT).

    python -m vision.imagenet32.prepare [--data_dir DIR] [--workers 32] [--force]

Idempotent: existing parquet files are not downloaded again and a complete cache is left
alone unless --force. Prints `PREPARE DONE ...` on success.
"""
import argparse
import io
import json
import os
import time
import urllib.request
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from vision.imagenet32.data import (DATA_ROOT, HF_URL, N_CLASSES, N_TRAIN, N_VAL,
                                    PARQUETS, cache_paths, cache_ready)


def download(name, dest):
    """Download the parquet file `name` to `dest` (via dest.part) unless it exists."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[prepare] have {dest} ({os.path.getsize(dest)/1e6:.0f} MB)", flush=True)
        return dest
    url = HF_URL.format(name=name)
    print(f"[prepare] GET {url}", flush=True)
    t0 = time.time()
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    print(f"[prepare]   -> {dest} ({os.path.getsize(dest)/1e6:.0f} MB, "
          f"{time.time()-t0:.0f}s)", flush=True)
    return dest


def _decode(b):
    im = Image.open(io.BytesIO(b))
    if im.mode != "RGB":
        im = im.convert("RGB")
    a = np.asarray(im, dtype=np.uint8)
    assert a.shape == (32, 32, 3), a.shape
    return a


def decode_split(paths, workers):
    """parquet files -> (N,32,32,3) uint8, (N,) int32 (labels 0..999)."""
    xs, ys = [], []
    with Pool(workers) as pool:
        for p in paths:
            t0 = time.time()
            pf = pq.ParquetFile(p)
            for rg in range(pf.num_row_groups):
                tb = pf.read_row_group(rg, columns=["image", "label"])
                blobs = [d["bytes"] for d in tb.column("image").to_pylist()]
                lab = np.asarray(tb.column("label").to_pylist(), dtype=np.int32)
                xs.append(np.stack(pool.map(_decode, blobs, chunksize=512)))
                ys.append(lab)
            print(f"[prepare] decoded {os.path.basename(p)}: "
                  f"{sum(len(y) for y in ys)} images so far ({time.time()-t0:.0f}s)",
                  flush=True)
    return np.concatenate(xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser(description="Download ImageNet32 and cache it as uint8 "
                                             "arrays for vision.train.")
    ap.add_argument("--data_dir", default=DATA_ROOT,
                    help="cache directory (default: data/imagenet32 in the repository, "
                         "or $IMAGENET32_ROOT)")
    ap.add_argument("--workers", type=int, default=32,
                    help="processes decoding the PNG images")
    ap.add_argument("--force", action="store_true",
                    help="rebuild the cache even if it is complete")
    args = ap.parse_args()

    root = os.path.expanduser(args.data_dir)
    raw = os.path.join(root, "parquet")
    os.makedirs(raw, exist_ok=True)
    P = cache_paths(root)
    if cache_ready(root) and not args.force:
        print(f"[prepare] cache already complete under {root}; nothing to do "
              f"(--force to rebuild)", flush=True)
        print("PREPARE DONE (cached)", flush=True)
        return

    files = {s: [download(n, os.path.join(raw, n)) for n in names]
             for s, names in PARQUETS.items()}

    xva, yva = decode_split(files["validation"], args.workers)
    assert len(yva) == N_VAL, len(yva)
    assert yva.min() >= 0 and yva.max() < N_CLASSES, (yva.min(), yva.max())
    np.save(P["val_x"], xva); np.save(P["val_y"], yva)
    print(f"[prepare] val: {xva.shape} -> {P['val_x']}", flush=True)

    xtr, ytr = decode_split(files["train"], args.workers)
    assert len(ytr) == N_TRAIN, len(ytr)
    assert ytr.min() >= 0 and ytr.max() < N_CLASSES, (ytr.min(), ytr.max())
    np.save(P["train_x"], xtr); np.save(P["train_y"], ytr)
    print(f"[prepare] train: {xtr.shape} -> {P['train_x']}", flush=True)

    # per-channel statistics on the [0, 1] scale, accumulated in float64 over chunks
    n = 0; s = np.zeros(3); s2 = np.zeros(3)
    for i in range(0, len(xtr), 65536):
        c = xtr[i:i + 65536].astype(np.float64) / 255.0
        n += c.shape[0] * 32 * 32
        s += c.sum(axis=(0, 1, 2)); s2 += (c * c).sum(axis=(0, 1, 2))
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)
    stats = {"mean": mean.tolist(), "std": std.tolist(), "n_train": int(len(ytr)),
             "n_val": int(len(yva)), "n_classes": N_CLASSES, "source": HF_URL,
             "class_hist_min": int(np.bincount(ytr, minlength=N_CLASSES).min()),
             "class_hist_max": int(np.bincount(ytr, minlength=N_CLASSES).max())}
    json.dump(stats, open(P["stats"], "w"), indent=2)
    print(f"[prepare] stats: mean={np.round(mean, 4).tolist()} "
          f"std={np.round(std, 4).tolist()} -> {P['stats']}", flush=True)
    print(f"PREPARE DONE train={xtr.shape} val={xva.shape} root={root}", flush=True)


if __name__ == "__main__":
    main()
