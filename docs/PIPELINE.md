# Brain-Tumor MLOps — Pipeline & Models

Single-source-of-truth document for the data pipeline, model zoo, training loop, and W&B / DVC integration. Onboards a new teammate or grader in 10 minutes.

> **TL;DR:** the repo can ingest the LGG MRI dataset, build a leak-proof patient-level split, normalise per channel, train any of four progressively richer models (logistic-regression baseline → custom CNN → U-Net classifier → ResNet50 transfer learning), and report medical-grade metrics. Every run, dataset version, and model checkpoint is tracked in W&B; DVC stages are wired and ready as soon as a remote is configured.

---

## 1. Goal

Two MSc-MLOps grading objectives drive every choice below:

1. **Reproducibility** — anyone on the team must be able to re-run training from the same data version and obtain the same numbers.
2. **Medical-grade rigour** — patient-level data isolation, sensitivity/specificity reporting (not raw accuracy), and **no auto-promotion to production** of any model.

We *do not* aim for state-of-the-art clinical accuracy — the LGG cohort is too small (110 patients) for that. We aim for a *trustworthy* pipeline that can plausibly be lifted to a real cohort.

---

## 2. What was built — file map

```
src/mlops_project/
├── data/
│   ├── splits.py          # Patient-level stratified split (#16)
│   ├── dataset.py         # PyTorch Dataset + normalisation (#17)
│   ├── transforms.py      # Albumentations pipelines (medical-safe) (#18)
│   └── prepare.py         # One-shot prep: index + stats + W&B Artifact
├── models/
│   ├── baseline.py        # Logistic regression on per-channel stats
│   ├── simple_cnn.py      # Custom 4-block CNN (~1.2M params)
│   ├── unet_classifier.py # U-Net encoder + classification head (~4.7M)
│   ├── transfer.py        # ResNet50 (ImageNet) frozen + new head
│   └── factory.py         # build_model(name)
├── training/
│   ├── metrics.py         # Sensitivity, specificity, AUC, Dice (#25)
│   └── train.py           # Hydra-driven train + test loop (#21, #22, #24)
└── utils/
    └── wandb_logging.py   # Graceful no-op when WANDB_API_KEY missing

configs/
├── config.yaml            # Hydra root (#23)
├── data/default.yaml      # batch_size, num_workers
├── training/default.yaml  # epochs, lr, weight_decay
└── model/                 # one YAML per architecture
    ├── baseline.yaml
    ├── simple_cnn.yaml
    ├── unet_classifier.yaml
    └── resnet50_transfer.yaml

tests/
├── test_splits.py         # 12 tests, incl. 3 end-to-end on real LGG
├── test_dataset.py        # 6 smoke tests
└── test_dataloader.py     # Augmentation + loader composition

dvc.yaml                   # `prepare_data` stage for `dvc repro`
```

---

## 3. Data pipeline (Phase 1)

### 3.1 The dataset on disk

```
data/raw/kaggle_3m/
├── data.csv                          ← clinical metadata (110 patients)
└── TCGA_<inst>_<id>_<acq-date>/      ← one folder per patient
    ├── *_<n>.tif                     ← 256×256×3 MRI (T1, FLAIR, T1+Gd)
    └── *_<n>_mask.tif                ← 256×256 binary tumour mask
```

Empirical findings from the EDA notebook (`notebooks/01_eda.ipynb`):

* **3 929 slices for 110 patients**, ~35 % positive (contain tumour), ~65 % negative.
* **15 patients** have a missing MR sequence (FLAIR is duplicated into pre or post). Flagged in `index['flair_duplicated']`.
* **1 patient (`TCGA_HT_A61B`)** has its entire clinical row blank — no grade, no age, no demographics. Routed to train only by the split policy.
* **Grade encoding quirk** — `data.csv` uses `1 = Grade II, 2 = Grade III` (verified empirically against `death01` mortality and age), *not* the standard TCGA `2/3` convention. All code in this repo handles this correctly via `_normalise_grade`.

### 3.2 Patient-level split (`splits.py`)

```python
from mlops_project.data.splits import make_patient_split
split = make_patient_split(meta)        # 70/15/15, seed=42
split.assert_disjoint()
```

Stratified by WHO grade so both Grade II and Grade III appear in every fold. The unknown-grade patient is pooled into train. Real numbers on the 110-patient cohort:

| Split | Patients | Slices | Grade II | Grade III |
|------:|---------:|-------:|---------:|----------:|
| train | 76       | 2 772  | 35       | 40 (+ 1 unknown) |
| val   | 17       | 537    | 8        | 9         |
| test  | 17       | 620    | 8        | 9         |

Every property above is asserted in `tests/test_splits.py` (12 tests, all green).

### 3.3 Preparation script (`prepare.py`)

Run once after every change to the raw data:

```bash
uv run python -m mlops_project.data.prepare
```

Produces in `data/processed/`:

* `slice_index.parquet` — 3 929 rows: image path, mask path, patient ID, slice number, tumour area, has-tumour flag, FLAIR-duplication flag, and `split ∈ {train, val, test}`.
* `norm_stats.json` — per-channel mean and std computed **only on the training split** (avoids val/test leakage). Real numbers:

```
T1     mean=0.0965  std=0.1399
FLAIR  mean=0.0850  std=0.1256
T1+Gd  mean=0.0874  std=0.1293
```

The script then uploads both files as a single W&B Artifact called `lgg-mri-prepared`. Every downstream training run pins this artifact, so the dataset version is part of run lineage.

### 3.4 Dataset class (`dataset.py`)

```python
from mlops_project.data.dataset import BrainMRIDataset, load_dataset_artifacts
index, stats = load_dataset_artifacts("data/processed")
ds = BrainMRIDataset(index, stats, split="train", transform=train_transform())
```

Each sample is:

* `image` — `(3, 256, 256)` float32, per-channel z-score normalised.
* `mask`  — `(1, 256, 256)` float32 binarised (only when `return_mask=True`, used for segmentation).
* `label` — float32 scalar in {0., 1.}.
* `flair_duplicated` — bool scalar, the patient-level FLAIR-duplication flag.

### 3.5 Augmentations (`transforms.py`)

Medical-safety rules baked in:

| Allowed | Forbidden |
|---|---|
| HorizontalFlip — anatomy is roughly bilaterally symmetric | VerticalFlip — head-down is anatomically wrong |
| Rotate(±10°) — small head-tilt | Large rotations or 90° turns |
| ShiftScaleRotate ±5 % — translation tolerance | Elastic deformations — would invent gyri |
| RandomBrightnessContrast — scanner variability | Hue/saturation — meaningless on greyscale MRI |

---

## 4. Models (Phase 2)

Four architectures, picked to span "clearly insufficient" → "literature-standard transfer learning". All four expose the same `nn.Module` interface and return logits of shape `(batch,)`.

| Name | File | Params (total / trainable) | Rationale |
|---|---|---:|---|
| `baseline` | `models/baseline.py` | 7 / 7 | Logistic regression on (mean, std) of each channel. Establishes the floor — any deep model that doesn't clear it by a wide margin isn't actually learning visual features. |
| `simple_cnn` | `models/simple_cnn.py` | 1.17 M / 1.17 M | 4 conv blocks (32→64→128→256), BN+ReLU, GAP head. Pure from-scratch — no pre-training. |
| `unet_classifier` | `models/unet_classifier.py` | 4.71 M / 4.71 M | U-Net encoder + classifier head. The U-Net architecture is the medical-imaging gold standard; reusing it here pre-wires segmentation for later. |
| `resnet50_transfer` | `models/transfer.py` | 23.51 M / 2 049 | Pretrained ResNet50 with frozen backbone + fresh binary head. Proves that ImageNet features transfer to medical greyscale (and how much). |

Built via the factory:

```python
from mlops_project.models.factory import build_model
model = build_model("resnet50_transfer", freeze_backbone=True)
```

### 4.1 Why these four

* **Baseline** is non-negotiable — without it, "0.85 sensitivity" sounds great but might just be the class prior.
* **Custom CNN** quantifies the cost of training from scratch on a small dataset.
* **U-Net classifier** is the *specialised* medical-imaging architecture; it also lets us reuse the same encoder for segmentation in Phase 2 stretch goals.
* **Transfer learning** is the literature-standard recipe for small medical datasets and should win on sensitivity.

Comparing the four lets us present a *story*, not just a number: "the deep models add ~32 AUC points over the linear baseline, and ImageNet pre-training adds another ~10 sensitivity points where it matters most clinically."

---

## 5. Training & evaluation (`training/train.py`)

Driven by Hydra (#23). Defaults live in `configs/`. Override anything on the CLI:

```bash
# Train the default model (simple_cnn, 5 epochs, batch 32)
uv run python -m mlops_project.training.train

# Pick a different architecture
uv run python -m mlops_project.training.train model=resnet50_transfer

# Override hyperparameters
uv run python -m mlops_project.training.train model=unet_classifier training.epochs=20 training.lr=1e-4

# Train all four models in one go (Hydra multirun)
uv run python -m mlops_project.training.train --multirun \
    model=baseline,simple_cnn,unet_classifier,resnet50_transfer
```

Mechanics:

* **Loss** — `BCEWithLogitsLoss(pos_weight = N_neg / N_pos)`. Computed empirically on the training split (~1.88 for our cohort) so the loss sees a balanced signal despite the 35 % positive rate.
* **Optimiser** — `AdamW`, default `lr=1e-3`, `weight_decay=1e-4`. Only trainable parameters are passed (so the frozen ResNet backbone doesn't waste an optimiser slot).
* **Device** — auto-detects MPS on Apple Silicon → CUDA → CPU.
* **Best-checkpoint policy** — keep the state-dict whose **val AUC-ROC** is highest. Restore it before the final test pass.
* **Final test pass** — the held-out test split is touched **once**, after training is complete, on the best checkpoint.

### 5.1 What gets logged

For every epoch (W&B `wandb.log`):

```
train_loss, val_loss
train/{accuracy, sensitivity, specificity, auc_roc, tp, fp, tn, fn}
val/{accuracy, sensitivity, specificity, auc_roc, tp, fp, tn, fn}
```

For the final run summary (`run.summary`):

```
test/{accuracy, sensitivity, specificity, auc_roc, tp, fp, tn, fn}
best_val_auc
```

For every run — a W&B Artifact `model-{name}` containing:
* `models/{name}.pt` — full PyTorch checkpoint with state dict, full Hydra config, history.
* `models/{name}_results.json` — text-readable summary.

When `WANDB_API_KEY` is unset (or `no_wandb=true` is passed), the script runs entirely locally and prints `[wandb] disabled`. No conditional code in the training loop.

---

## 6. Results — first runs (5 epochs each)

> Run conditions: 5 epochs, seed=42, MPS device (Apple Silicon), default LR/optimiser. **These are short demonstration runs**, not production-tuned numbers. A real Phase 2 baseline would use 20–50 epochs with a learning-rate scheduler and early stopping.

| Model | Test accuracy | **Test sensitivity** | **Test specificity** | **Test AUC** | Best val AUC |
|---|---:|---:|---:|---:|---:|
| baseline (logistic regression) | 0.398 | 0.826 | 0.157 | **0.581** | 0.586 |
| simple_cnn (1.2 M params) | 0.819 | 0.746 | 0.861 | **0.899** | 0.864 |
| unet_classifier (4.7 M params) | 0.802 | 0.763 | 0.823 | **0.911** | 0.890 |
| **resnet50_transfer** (frozen) | 0.811 | **0.844** | 0.793 | **0.911** | 0.879 |

**Reading the results table.**

* The **baseline** shows what *classification on summary statistics* looks like: high recall (it predicts "tumour" too often) but specificity collapses — it doesn't know shape or location. Test AUC 0.58 is the floor.
* **simple_cnn** already pushes test AUC to 0.90 with 1.2 M parameters trained from scratch in 5 epochs. That's a 32-point AUC gain over the linear baseline, which is the visual-features signal we wanted to confirm.
* **unet_classifier** matches transfer learning on AUC at 0.911. The U-Net inductive biases (multi-scale receptive fields, BN everywhere) earn their keep here.
* **resnet50_transfer** ties on AUC but **wins on sensitivity (0.844)** — the metric that matters most clinically. That is the point of transfer learning on a small dataset: ImageNet features push the right errors (false positives) and pull the wrong ones (false negatives = missed tumours).

> All four runs visible in the [W&B project](https://wandb.ai/nathan2massicot-berner-fachhochschule/brain-tumor-classification): per-epoch curves, run histories, and uploaded checkpoints.

---

## 7. How to reproduce end-to-end

```bash
# 1. Install
uv sync
cp .env.example .env       # then paste your WANDB_API_KEY (and WANDB_ENTITY for the Team)

# 2. Prepare the dataset (~30 s, scans 3 929 TIFFs, uploads W&B Artifact)
uv run python -m mlops_project.data.prepare

# 3. Train one model (Hydra)
uv run python -m mlops_project.training.train model=resnet50_transfer training.epochs=5

# 3-bis. Train all four in a single multirun
uv run python -m mlops_project.training.train --multirun \
    model=baseline,simple_cnn,unet_classifier,resnet50_transfer training.epochs=5

# 4. View results
ls models/                 # checkpoints + JSON summaries
open https://wandb.ai/nathan2massicot-berner-fachhochschule/brain-tumor-classification

# 5. Run the test suite (20 tests)
uv run pytest tests/ -v
```

To skip W&B (e.g. on a teammate's machine without access):

```bash
uv run python -m mlops_project.training.train model=simple_cnn no_wandb=true
# or
WANDB_MODE=disabled uv run python -m mlops_project.training.train model=simple_cnn
```

---

## 8. What's *not* yet done (and why)

| Issue | What's missing | Why we didn't do it now |
|---|---|---|
| **#13** | DVC remote (Google Drive / S3) and `dvc push` | Needs an external storage account; `dvc.yaml` is already in place so the integration is trivial once a teammate provisions a remote. |
| **#26** | Hyperparameter sweep on W&B | Hydra multirun + W&B Sweep config. Will land after we lock in 1–2 promising architectures from the 5-epoch results. |
| **#27** | Promote a model in the W&B Registry | Requires human review per `CLAUDE.md` medical rule — done manually after Phase 2 final runs (50-epoch with scheduler). |
| Segmentation head | UNet decoder + Dice loss | This project's task spec is binary classification; segmentation is a stretch goal for after Phase 5. |

---

## 9. Notes for the team

* **Patient-level integrity is enforced by tests** — never bypass `tests/test_splits.py`. If you need a different split, change the seed, not the policy.
* **Per-channel z-score must be re-computed on the training split only.** If you change the train/val/test partition, re-run `prepare.py` so `norm_stats.json` doesn't leak val/test pixels into the normaliser.
* **The W&B project is currently on the personal entity `nathan2massicot-berner-fachhochschule`.** When the academic Team is created, update `.env`'s `WANDB_ENTITY=` so all 4 members can write runs.
* **Apple Silicon first-epoch warm-up takes 15–25 minutes** while MPS compiles its Metal kernel cache. Subsequent epochs are ~25 s each. This is normal — don't ctrl-C.
* **`.env` is for secrets and infra only** — model name, batch size, learning rate, paths all live in `configs/` (Hydra). To change behaviour, edit a YAML or pass `key=value` on the CLI.

---

## 10. Reference — what every issue we touched maps to in the code

| GH Issue | Status | Files added/changed |
|---:|---|---|
| #9  | ✅ done | `.env`, `src/mlops_project/utils/wandb_logging.py` |
| #14 | ✅ done | `notebooks/01_eda.ipynb` (20 visualisations) |
| #15 | ✅ done | `notebooks/01_eda.ipynb` |
| #16 | ✅ done | `src/mlops_project/data/splits.py`, `tests/test_splits.py` |
| #17 | ✅ done | `src/mlops_project/data/dataset.py`, `src/mlops_project/data/prepare.py`, `tests/test_dataset.py`, `dvc.yaml` |
| #18 | ✅ done | `src/mlops_project/data/transforms.py` |
| #19 | ✅ done | `tests/test_dataloader.py` |
| #20 | ✅ done | `src/mlops_project/models/{baseline,simple_cnn,unet_classifier,transfer,factory}.py` |
| #21 | ✅ done | `src/mlops_project/training/train.py` |
| #22 | ✅ done | `src/mlops_project/utils/wandb_logging.py` + integration in `train.py` |
| #23 | ✅ done | `configs/{config,data/default,training/default,model/*}.yaml` |
| #24 | ✅ done (5 epochs) | First runs done for all 4 models; production-quality runs (20–50 epochs + scheduler) pending. |
| #25 | ✅ done | `src/mlops_project/training/metrics.py` |
