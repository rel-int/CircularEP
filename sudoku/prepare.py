"""Prepare the Sudoku data: convert the "sudoku-hard" benchmark of Palm et al. 2018
(Recurrent Relational Networks; 180k train / 18k valid / 18k test boards, 17-34 givens)
into the split files read by sudoku.data.

    python -m sudoku.prepare [--data_dir data/sudoku] [--zip PATH]

Without --zip the official archive is downloaded (the authors' link, then a mirror) and
cached as <data_dir>/sudoku-hard.zip. The archive holds train.csv / valid.csv /
test.csv, one "puzzle,solution" row per board, each an 81-digit row-major string with
0 = blank. Output: <data_dir>/{train,valid,test}.npz, each with keys `puzzles` and
`solutions`, uint8, shape (n, 81), in the row order of the CSV files. The training and
evaluation entry points read this directory by default (or $SUDOKU_RRN_DIR / their
--data_dir).
"""
import argparse
import hashlib
import os
import urllib.request
import zipfile

import numpy as np

from .data import RRN_DIR

URL = "https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1"
MIRROR = "https://data.dgl.ai/dataset/sudoku-hard.zip"
SHA256 = "99f5c1f3f9a7c26e2e52d087dba2b312a816c9a54c638811455db72b8d3aa30d"
ROWS = {"train": 180000, "valid": 18000, "test": 18000}
N_CELLS = 81


def download(path):
    """Download the archive to `path` (authors' link first, then the mirror)."""
    errors = []
    for url in (URL, MIRROR):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                data = r.read()
            break
        except Exception as e:
            errors.append(f"{url}: {e}")
    else:
        raise RuntimeError("could not download the benchmark:\n" + "\n".join(errors))
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def parse_csv(text):
    """'puzzle,solution' rows -> (puzzles, solutions), uint8 (n, 81)."""
    rows = [ln.strip() for ln in text.split("\n") if ln.strip()]
    flat = np.frombuffer("".join(ln.replace(",", "") for ln in rows).encode(), np.uint8)
    flat = (flat - ord("0")).reshape(len(rows), 2 * N_CELLS)
    return flat[:, :N_CELLS].copy(), flat[:, N_CELLS:].copy()


def read_zip(path):
    """{split: csv text} from the archive (members are matched by basename)."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base in {f"{s}.csv" for s in ROWS}:
                out[base[:-4]] = z.read(name).decode()
    missing = set(ROWS) - set(out)
    if missing:
        raise RuntimeError(f"{path} lacks {sorted(missing)}.csv")
    return out


def check(split, puz, sol):
    """Row count, digit ranges, givens range, clues consistent with solution."""
    assert len(puz) == ROWS[split], f"{split}: {len(puz)} rows, expected {ROWS[split]}"
    assert puz.max() <= 9 and sol.min() >= 1 and sol.max() <= 9, f"{split}: bad digits"
    givens = (puz > 0).sum(axis=1)
    assert givens.min() >= 17 and givens.max() <= 34, f"{split}: givens out of 17-34"
    assert np.all((puz == 0) | (puz == sol)), f"{split}: clue differs from solution"


def convert(zip_path, out_dir):
    """Write {train,valid,test}.npz under out_dir from the archive at zip_path."""
    os.makedirs(out_dir, exist_ok=True)
    for split, text in read_zip(zip_path).items():
        puz, sol = parse_csv(text)
        check(split, puz, sol)
        path = os.path.join(out_dir, f"{split}.npz")
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, puzzles=puz, solutions=sol)
        os.replace(tmp, path)
        print(f"{path}: {puz.shape[0]} boards, mean givens {(puz > 0).sum(1).mean():.2f}",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=RRN_DIR,
                    help="output directory (default: data/sudoku in the repository, or "
                         "$SUDOKU_RRN_DIR)")
    ap.add_argument("--zip", default=None,
                    help="local sudoku-hard.zip (default: download it into --data_dir)")
    args = ap.parse_args()
    path = args.zip
    if path is None:
        os.makedirs(args.data_dir, exist_ok=True)
        path = os.path.join(args.data_dir, "sudoku-hard.zip")
        if not os.path.exists(path):
            download(path)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest != SHA256:
        print(f"warning: archive sha256 {digest} differs from the reference {SHA256}",
              flush=True)
    convert(path, args.data_dir)


if __name__ == "__main__":
    main()
