# Circular Predictive Coding Unlocks Effective Depth

Closing a shallow feedforward predictive coding network into a loop turns relaxation time into effective depth; the model is trained with equilibrium propagation (EP), and every weight update is local.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/built%20with-JAX-A142F4)](https://github.com/jax-ml/jax)
[![Live demo](https://img.shields.io/badge/live%20demo-GitHub%20Pages-2EA44F)](https://rel-int.github.io/CircularEP/)

Code for the paper *Circular Predictive Coding Unlocks Effective Depth* by Tommaso Salvatori and Matteo Forasassi (Relational Intelligence, Inc. and TU Wien).

![Animation of a feedforward predictive coding network whose first and last hidden layers are identified, so that its layers close into a loop.](assets/circular_model.gif)

*A feedforward predictive coding network closed into a loop.*

Take a feedforward predictive coding network and identify its first and last hidden layers, $x_0 \equiv x_L$, so that the layers form a loop. The relaxation of the value nodes then sweeps around the loop instead of stopping at the output, so the effective depth of the network is set by the relaxation time rather than by the number of layers. The free equilibrium of the loop is the fixed point of an equivalent deep equilibrium model (Theorem 1).

![Animation of the three learning phases on a 4×4 Sudoku: the value nodes of a four-layer loop relax while their prediction errors and the energy shrink (free phase), the output is pulled towards the label (nudging), and the weights are updated (weight update).](assets/learning_phases.gif)

*The three phases of learning: free phase, nudged phase and local weight update.*

1. **Free phase.** Inference relaxes the value nodes to the zero-energy equilibrium $x^*$.
2. **Nudged phase.** The output is weakly pulled towards the label with strength $\beta$, and the network settles at a nearby equilibrium $x^\beta$. The computer vision model runs one nudged phase per training step, with a random sign of $\beta$; the Sudoku models run a $+\beta$ and a $-\beta$ phase and average the two updates.
3. **Local weight update.** $\Delta\theta = -\frac{\eta}{\beta}\frac{\partial E}{\partial\theta}(\theta, x^{\beta})$, with $\beta$ signed. Prediction errors vanish at $x^*$, so the free-phase term drops out, and each weight is updated from the prediction error of the node it projects to and the activity of the node it reads from.

EP approximates the implicit gradient of the equivalent deep equilibrium model (Theorem 2).

## Results

Mean ± s.d. over independent training runs. Computer vision: best top-1 accuracy over epochs (%), on the test set for CIFAR-10 and CIFAR-100 and on the 50,000-image validation set for ImageNet32. Sudoku: board accuracy (%) on the 18,000 test boards of Sudoku-hard.

### Computer vision

A circular convolutional model with a value node after every layer, encoder included, so every weight is updated from the prediction error of the node it projects to.

| Dataset | Accuracy | Command |
|---|---|---|
| CIFAR-10 | 93.54 ± 0.11 | `python -m vision.train --dataset cifar10` |
| CIFAR-100 | 72.38 ± 0.32 | `python -m vision.train --dataset cifar100` |
| ImageNet32 | 39.31 ± 0.24 | `python -m vision.train --dataset imagenet32` |

### Sudoku

- **Standard:** a circular MLP-Mixer with 1.1M parameters, read out after 300 sweeps.
- **Recursive:** a circular MLP-Mixer with 105k parameters, re-run for 32 stages, each starting from the previous stage's answer, and trained with deep supervision.

Views are digit relabelings of the same board; among 64 views, the answer with the lowest self-cost (the model's own readout cost, no labels) is selected.

| Model | One view | 64 views |
|---|---|---|
| Standard | 66.4 ± 3.4 | 90.1 ± 1.1 |
| Recursive | 84.4 ± 4.2 | 98.2 ± 0.7 |

## Interactive demo

The [live demo](https://rel-int.github.io/CircularEP/) runs a small trained circular network (two layers of 256 tanh units joined into a loop) on 4×4 Sudoku in the browser. It relaxes the energy live, folds and unfolds the loop into the equivalent deep feedforward network, applies the $+\beta$ nudge, shows the EP weight update, and lets you edit the clues. The demo is a static page with no build step; to run it locally:

```bash
cd demo
python -m http.server 8000   # then open http://localhost:8000
```

Opening `demo/index.html` directly from disk does not work, because browsers block ES modules on `file://`. See [demo/README.md](demo/README.md) for details.

## Installation

Python 3.11+ (tested with 3.12). The models are written in JAX. On a machine with an NVIDIA GPU, install the CUDA build of JAX first, then the remaining requirements:

```bash
git clone https://github.com/rel-int/CircularEP.git
cd CircularEP
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -U "jax[cuda12]==0.4.38"   # NVIDIA GPU only; skip on CPU-only machines
pip install -r requirements.txt
```

On a CPU-only machine, `pip install -r requirements.txt` installs the CPU build of JAX, which is enough for the smoke test. `torch` and `torchvision` are only used to download CIFAR-10.

## Data

All data lives under `data/` by default:

- **CIFAR-10 and CIFAR-100** are downloaded automatically into `data/cifar` on first use.
- **ImageNet32:** `python -m vision.imagenet32.prepare` downloads the dataset and builds the arrays in `data/imagenet32`.
- **Sudoku-hard** (Palm et al., 2018; 180,000 training, 18,000 validation and 18,000 test boards): `python -m sudoku.prepare` downloads and converts the dataset into `data/sudoku`.

Every script accepts `--data_dir`; the environment variables `IMAGENET32_ROOT` and `SUDOKU_RRN_DIR` change the ImageNet32 and Sudoku locations.

## Quick start

Run everything from the repository root. Outputs are written to `runs/`, in a folder named after the run's `--seed` (default 0).

```bash
# Computer vision (writes runs/vision/<dataset>/s0/)
python -m vision.train --dataset cifar10
python -m vision.train --dataset cifar100
python -m vision.imagenet32.prepare            # once
python -m vision.train --dataset imagenet32

# Sudoku data, once
python -m sudoku.prepare

# Standard Sudoku model: train, then evaluate with one view and 64 views
python -m sudoku.train                          # writes runs/sudoku/s0/
python -m sudoku.evaluate --ckpt runs/sudoku/s0/best_eq.pkl --views 64

# Recursive Sudoku model: train, then evaluate at 32 stages with one view and 64 views
python -m recursive.train                       # writes runs/recursive/s0/
python -m recursive.evaluate --ckpt runs/recursive/s0/best.pkl
python -m recursive.evaluate --ckpt runs/recursive/s0/best.pkl --views 64
```

## Smoke test

A short end-to-end check of every training and evaluation script on tiny settings (it needs CIFAR-10, downloaded automatically, and the Sudoku data from `python -m sudoku.prepare`):

```bash
python -m vision.train --dataset cifar10 --smoke --out runs/smoke/vision
python -m sudoku.train --epochs 1 --bs 4 --max_steps 2 --n_valid 8 --out runs/smoke/sudoku
python -m sudoku.evaluate --ckpt runs/smoke/sudoku/best_eq.pkl --views 2 --n 16
python -m recursive.train --out runs/smoke/recursive --set epochs=1 ep_size=16 bs=8 n_valid=16
python -m recursive.evaluate --ckpt runs/smoke/recursive/best.pkl --stages 4 --n 16
```

## Repository layout

- `core/`: the EP engine: free and nudged phases, fixed-point solvers, spectral diagnostics and training utilities.
- `vision/`: the circular convolutional model, CIFAR data loading, the CIFAR recipe (`recipes.py`) and the training script of every computer vision result (`train.py`).
- `vision/imagenet32/`: ImageNet32 download and cache (`prepare.py`), loader and recipe (`recipe.py`).
- `sudoku/`: the standard Sudoku model, data preparation, training (`train.py`) and evaluation with digit-relabeling views (`evaluate.py`).
- `recursive/`: the recursive Sudoku model, its training pipeline (`pipeline.py`, `train.py`) and evaluation (`evaluate.py`).
- `paper/`: the reproduction guide (`README.md`) and the per-view evaluation pipeline of the Sudoku table (`paper/sudoku/`).
- `demo/`: the interactive in-browser demo, deployed to GitHub Pages.
- `assets/`: the animations of this file.

## Reproducing the paper

[paper/README.md](paper/README.md) gives the exact commands, outputs, expected numbers and approximate compute for Table 1 and Table 4 (computer vision) and Table 5 and Table 6 (Sudoku).

## Citation

```bibtex
@misc{salvatori2026circular,
  title  = {Circular Predictive Coding Unlocks Effective Depth},
  author = {Salvatori, Tommaso and Forasassi, Matteo},
  year   = {2026}
}
```

The same metadata is in [CITATION.cff](CITATION.cff), which GitHub shows as "Cite this repository".
