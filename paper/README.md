# Reproducing the paper

All commands are run from the repository root (see the top-level [README.md](../README.md) for installation and data preparation). Training outputs go to `runs/`, derived data to `results/`; both are git-ignored.

The tables report the mean ± s.d. (ddof = 1) over independent training runs. Each training command trains one run; train further runs with other `--seed` values (the default output folder is named after the seed, `s<seed>`). Expect results that match the paper statistically, not bit for bit.

**Evaluation protocol.**

- Computer vision: the reported number is the best top-1 accuracy over epochs, evaluated after every epoch on the test set (CIFAR-10, CIFAR-100) or on the 50,000-image validation set (ImageNet32, whose test split is unlabeled). No part of the training set is held out.
- Sudoku: checkpoints are selected on the first 2,000 validation puzzles; all reported numbers are board accuracies on the 18,000 test boards.

## Table 1 and Table 4: computer vision

All computer vision results come from `vision.train`: the circular convolutional model with a value node after every layer, encoder included, so every weight is updated from the prediction error of the node it projects to. Every training step runs one nudged phase with a random sign of β. The presets of `vision.train` (`PRESETS`) hold the settings of each dataset on top of the recipe it names (Table 4): `RECIPE` in `vision/recipes.py` for both CIFAR datasets and `RECIPE` in `vision/imagenet32/recipe.py` for ImageNet32.

- CIFAR-10: CIFAR recipe (180 epochs, batch 64), 40 damped nudging sweeps (damping 0.3).
- CIFAR-100: CIFAR recipe (180 epochs, batch 64), 20 plain nudging sweeps.
- ImageNet32: ImageNet32 recipe (50 epochs, batch 128), 10 plain nudging sweeps, β halved (`beta_scale` 0.5).

```bash
python -m vision.train --dataset cifar10
python -m vision.train --dataset cifar100
python -m vision.imagenet32.prepare              # once: builds data/imagenet32
python -m vision.train --dataset imagenet32
```

Each run writes to `runs/vision/<dataset>/s<seed>/` (`--out`): the history `<dataset>_s<seed>.json` (`best_acc`, per-epoch `hist`), the best parameters `<dataset>_s<seed>_best.pkl` and the resume state `<dataset>_s<seed>_ckpt.pkl` (`--ckpt_dir` moves the two `.pkl` files); `--resume` continues an interrupted run. The accuracy `acc` is evaluated after every epoch with the Anderson-accelerated free phase on CIFAR (the plain-sweep accuracy is logged as `acc_plain`) and with 40 plain sweeps on ImageNet32 (the Anderson accuracy is logged as `acc_aa`); `best_acc` is its maximum over epochs. `--damp`, `--cT` and `--beta_scale` override the preset.

Expected best top-1 accuracy (%):

| Dataset | Accuracy |
|---|---|
| CIFAR-10 | 93.54 ± 0.11 |
| CIFAR-100 | 72.38 ± 0.32 |
| ImageNet32 | 39.31 ± 0.24 |

## Table 5 and Table 6: Sudoku-hard

Data: `python -m sudoku.prepare` (writes `data/sudoku`).

**Table 5: training recipes.** Table 5 lists the training recipes of the two Sudoku models. In the code they are `RECIPE` in `sudoku/train.py` (standard model) and `CONFIG` in `recursive/train.py` (recursive model; `CONFIG` is `DEFAULTS` of `recursive/pipeline.py` without the grid size and the seed). Both models run a +β and a −β nudged phase and average the two updates. An epoch is 8,000 training puzzles for both models, not a pass over the 180,000 training boards.

- Standard model: circular MLP-Mixer with 1.1M parameters, 250 epochs (batch 256). The checkpoint is selected on the damped readout (300 sweeps).
- Recursive model: circular MLP-Mixer with 105k parameters, 90 epochs (batch 128) with deep supervision: every batch runs 4 to 16 loops, one optimiser step each, and carries the answer from loop to loop. The checkpoint (EMA weights) is selected at 16 stages.

```bash
python -m sudoku.train          # standard model, writes runs/sudoku/s<seed>/
python -m recursive.train       # recursive model, writes runs/recursive/s<seed>/
```

`sudoku.train` writes `best_canon.pkl`, `best_eq.pkl` and `final.pkl`; the paper uses `best_eq.pkl` (selected on the damped equilibrium readout). `recursive.train` writes `best.pkl`, `last.pkl` and `log.json`; `--set KEY=VALUE` overrides a key of `CONFIG`.

**Table 6: test-time views.** One view and 64 digit-relabeled views of all 18,000 test boards, at 300 sweeps of the damped readout for the standard model and at 32 stages for the recursive model. With 64 views, the self-cost selection picks the view whose answer has the lowest readout cost of the model (no labels); the oracle counts a board as solved if any view solves it (a ceiling, it reads the labels).

```bash
python -m sudoku.evaluate --ckpt runs/sudoku/s0/best_eq.pkl --views 64
python -m recursive.evaluate --ckpt runs/recursive/s0/best.pkl
python -m recursive.evaluate --ckpt runs/recursive/s0/best.pkl --views 64
```

- `sudoku.evaluate` prints the single-view accuracy (`single board=`) and, over the 64 views, the self-cost selection (`cost=`) and the oracle (`oracle=`); `--views 1` gives the single view alone.
- `recursive.evaluate` with one view prints the board accuracy at several stages; the row `k = 32` is the single-view number. With `--views 64` it writes the per-view results and `summary.txt` to `runs/recursive/s0/best_tta/` (`--out_dir` changes it): `cost_last` is the self-cost selection, `oracle_last` the oracle and `view0_last` the single view, all at the last stage. The free-phase budget is stored in the checkpoint: 600 iterations at the first stage and 80 at each later stage (`--t_free` and `--t_warm` override it).

The same numbers, with the per-view results kept on disk, come from the `paper.sudoku` pipeline. Run `paper.sudoku.views` once per checkpoint and give every checkpoint of a model its own `--seed` label: the label names the output files, and existing files are skipped, so two checkpoints with the same label would silently share one result.

```bash
python -m paper.sudoku.views --model standard  --ckpt runs/sudoku/s0/best_eq.pkl --seed 0
python -m paper.sudoku.views --model standard  --ckpt runs/sudoku/s1/best_eq.pkl --seed 1
python -m paper.sudoku.views --model recursive --ckpt runs/recursive/s0/best.pkl --seed 0
python -m paper.sudoku.views --model recursive --ckpt runs/recursive/s1/best.pkl --seed 1
python -m paper.sudoku.analyze table6
```

- `paper.sudoku.views` defaults to the Table 6 settings (`--views 64`, `--sweeps 300` for the standard model, `--stages 32` for the recursive model, `--name table6`) and writes `results/sudoku/<model>_table6/views/s<seed>_v<a>_<b>.npz` (`--out_dir` changes the folder). It is resumable: repeating a command skips the files that exist, and `--v_start` / `--v_end` split the views over processes.
- `paper.sudoku.analyze table6` reads `results/sudoku/standard_table6` and `results/sudoku/recursive_table6` (`--std_dir`, `--rec_dir`) at sweep 300 and stage 32 (`--std_at`, `--rec_at`) and prints one view, self-cost selection and oracle as mean ± s.d. over the labels, followed by the per-label values.
- `paper.sudoku.measure` takes the same model flags and stores the single-view prediction after every sweep or stage in `results/sudoku/<model>_<name>/meas_s<seed>.npz`.

View 0 is the board itself for both models, but the standard and the recursive model draw their other views with different random generators (`view_perms` in `sudoku/evaluate.py`, `build_perms` in `recursive/data.py`), so the two models see different view sets. Each evaluation script uses the same generator as the pipeline for its model.

Expected board accuracy (%):

| Model | One view | 64 views, self-cost | 64 views, oracle |
|---|---|---|---|
| Standard, 300 sweeps | 66.4 ± 3.4 | 90.1 ± 1.1 | 90.3 ± 1.0 |
| Recursive, 32 stages | 84.4 ± 4.2 | 98.2 ± 0.7 | 98.2 ± 0.6 |

## Compute

Approximate wall-clock time on one H100:

| Experiment | Training (one run) | Evaluation |
|---|---|---|
| CIFAR-10, CIFAR-100 | 1.5–2 h | included in training |
| ImageNet32 | about 5 h | included in training |
| Sudoku, recursive model | about 7 h | under 1 h for one view; of the order of a day for 64 views |

Evaluating 64 views of the recursive model is the heaviest step: it costs 64 times the single-view evaluation.
