# Brain Tumor MLOps — Detection & Localization

End-to-end MLOps pipeline for **brain tumor detection and localization** from MRI images. The project covers the full ML lifecycle: data ingestion, training, inference API, frontend, monitoring, drift detection, and continuous retraining.

> ⚠️ **Disclaimer**: this is an **academic / educational project**. It is **not intended for clinical use**. No medical validation, no CE/FDA certification.

---

## Context

- **Course**: MSc-level MLOps module — graded project requiring at least 3 MLOps tools.
- **Team**: Gabriel Gillmann · Helena Martínez Río · Nathan Massicot · Jahnavi Patil.
- **Dataset**: [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (Kaggle).
- **Task**: binary classification (tumor / no tumor) on MRI slices, with localization as a stretch goal.
- **Model**: CNN with **transfer learning** (ResNet50, EfficientNet, etc.).

📘 **Full technical documentation**: [`docs/PIPELINE.md`](docs/PIPELINE.md) — data pipeline, models, results, architecture choices.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Env & dependency management | **uv** (Astral) |
| ML framework | **PyTorch** + torchvision |
| Experiment tracking | **Weights & Biases** |
| Configs | **Hydra** / OmegaConf |
| Data & model versioning | **DVC** + **W&B Artifacts** |
| API | **FastAPI** |
| Frontend | **Streamlit** |
| Containers | **Docker** + docker-compose |
| Monitoring | **Prometheus** + **Grafana** |
| Drift detection | **Evidently AI** |
| Orchestration / retraining | **Prefect** |
| Model Registry | **W&B Model Registry** |
| Tests | **pytest** + httpx |


---

## Prerequisites

- **Python 3.12**
- **[uv](https://docs.astral.sh/uv/)** — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Git** + **Git LFS** — `brew install git-lfs && git lfs install` (macOS) — required to fetch the model checkpoints in `models/*.pt`
- A free **Weights & Biases** account — https://wandb.ai
- A **Kaggle** account (only to download the raw dataset if you skip DVC)
- **Docker** (optional, for the full local stack)
- **GPU** recommended for training (CUDA / MPS); inference runs on CPU

---

## Setup for a new teammate — 5 steps

> Goal: from "I just cloned the repo" to "I can run a training and see live charts on W&B" in under 10 minutes.

### 1. Clone and install

```bash
git clone https://github.com/Nathan-massicot/BrainTumor_MLOps.git
cd BrainTumor_MLOps
uv sync
```

`uv sync` creates `.venv/` and installs **all** dependencies (prod + dev) from `pyproject.toml` / `uv.lock`. No need to `pip install` anything else.

> The four trained checkpoints (`models/*.pt`, ~115 MB total) are tracked via **Git LFS**, so `git clone` pulls them if you ran `git lfs install` once on your machine. If you cloned before installing LFS, run `git lfs pull` from inside the repo.
>
> If you want a **no-LFS path** (recommended when disk or LFS setup is problematic), skip `git lfs pull` and download real checkpoints from W&B artifacts in step 3.

### 2. Configure your `.env` (secrets and infra only)

```bash
cp .env.example .env
```

Edit `.env` and fill in the required values — most importantly `WANDB_API_KEY` and `WANDB_ENTITY=nathan2massicot-berner-fachhochschule` (only needed if you run trainings or pull from W&B; you can skip this if you only plan to **load** the existing checkpoints — see step 4).

### 3. Get the data and the trained model weights

You need two artefact sets in your local checkout:
- `data/processed/` — `slice_index.parquet` + `norm_stats.json` (used by `BrainMRIDataset`)
- `models/` — `*.pt` checkpoints (each one bundles the state_dict, the architecture name, and the full Hydra config)

The model weights come automatically with `git clone` (via Git LFS). The processed dataset is not in Git — pick whichever channel is easiest. **Loading a model afterwards does not require W&B** — see step 4.

| What | Simplest option | MLOps option |
|---|---|---|
| **Raw dataset** (original TIFFs, ~140 MB) | `uv run kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p data/raw --unzip` (needs Kaggle creds in `.env`) | `dvc pull` once #13 lands |
| **Prepared dataset** (`data/processed/`) | Ask a teammate to share `data/processed/` (~5 MB, zips well) — drop it in place | `uv run wandb artifact get nathan2massicot-berner-fachhochschule/brain-tumor-classification/lgg-mri-prepared:latest --root data/processed` |
| **Trained model weights** (`models/*.pt`, ~115 MB total) | Pulled by `git clone` only if **Git LFS** is installed. If needed: `git lfs install` then `git lfs pull`. | Pull directly from W&B (no LFS needed) using the commands below. |

**No-LFS model download (real checkpoints from W&B)**

Use this path if Git LFS is not available or your machine has LFS temp-space issues.

```bash
uv run wandb artifact get nathan2massicot-berner-fachhochschule/brain-tumor-classification/model-baseline:latest --root models
uv run wandb artifact get nathan2massicot-berner-fachhochschule/brain-tumor-classification/model-simple_cnn:latest --root models
uv run wandb artifact get nathan2massicot-berner-fachhochschule/brain-tumor-classification/model-unet_classifier:latest --root models
uv run wandb artifact get nathan2massicot-berner-fachhochschule/brain-tumor-classification/model-resnet50_transfer:latest --root models
```

These are the same real `.pt` checkpoints used by the API/frontend stack, not synthetic placeholders.

> If you don't want any download at all and have a GPU/MPS handy: retrain everything in ~30 min with `uv run python -m brain_tumor_mlops.training.train --multirun model=baseline,simple_cnn,unet_classifier,resnet50_transfer`.

**Sanity check** — after `git clone`, confirm the LFS files came through:

```bash
ls -lh models/*.pt
# Expected: ~5 KB, ~4.5 MB, ~18 MB, ~90 MB. If everything shows ~130 B, you cloned without LFS — run `git lfs pull`.
```

The `.pt` files are stored via **Git LFS**, not as raw Git blobs, so the repo itself stays light while `git clone` still ends up with the files in `models/`.

### 4. Reuse a model locally — no W&B, no retraining

Once a `.pt` checkpoint is in `models/`, three lines load it back into a ready-to-use `nn.Module`. **No W&B login or network call is involved** — `load_checkpoint()` is a pure local-file reader:

```python
import torch
from brain_tumor_mlops.models.factory import load_checkpoint
from brain_tumor_mlops.data.dataset import BrainMRIDataset, load_dataset_artifacts
from brain_tumor_mlops.data.transforms import eval_transform

# 1. Rebuild model + load weights (architecture is read from the checkpoint)
model, ckpt = load_checkpoint("models/resnet50_transfer.pt", device="cpu")
print(f"loaded {ckpt['model_name']} — test AUC={ckpt['test_metrics']['auc_roc']:.3f}")

# 2. Run a prediction on any prepared test slice
index, stats = load_dataset_artifacts("data/processed")
ds = BrainMRIDataset(index, stats, split="test", transform=eval_transform())
sample = ds[0]

with torch.no_grad():
    logit = model(sample["image"].unsqueeze(0))
    prob = torch.sigmoid(logit).item()

print(f"P(tumour)={prob:.3f}, ground-truth={sample['label'].item()}")
```

`load_checkpoint()` reads the architecture name and its kwargs from the `.pt` file itself, so you don't need to remember whether the file holds a SimpleCNN or a ResNet50 — the call works the same.

The returned `ckpt` dict also exposes:
- `ckpt['best_val_auc']` — the val AUC at the saved epoch
- `ckpt['test_metrics']` — `{accuracy, sensitivity, specificity, auc_roc, tp, fp, tn, fn}`
- `ckpt['history']` — per-epoch metrics, useful to plot training curves locally
- `ckpt['hydra_cfg']` — the full Hydra config used (model kwargs, lr, epochs, batch size, seed) so the run is fully reproducible

To predict on **your own** image (not from the prepared test set), apply the same per-channel z-score normalisation the Dataset uses — the stats live in `data/processed/norm_stats.json`. Easiest pattern: instantiate `BrainMRIDataset` once and let it handle the preprocessing.

### 5. Verify everything works

```bash
uv run pytest tests/ -v        # 20 tests, must be 100% green
```

Green → you're ready to train.

---

## Daily workflow

### Run a training

```bash
# Default model (simple_cnn, 5 epochs, batch 32)
uv run python -m brain_tumor_mlops.training.train

# Pick one of the 4 architectures
uv run python -m brain_tumor_mlops.training.train model=resnet50_transfer
uv run python -m brain_tumor_mlops.training.train model=unet_classifier

# Override hyperparameters from the CLI (Hydra syntax)
uv run python -m brain_tumor_mlops.training.train model=resnet50_transfer training.epochs=20 training.lr=1e-4 data.batch_size=64

# Train all 4 models in a row (multirun)
uv run python -m brain_tumor_mlops.training.train --multirun \
    model=baseline,simple_cnn,unet_classifier,resnet50_transfer training.epochs=10
```

Every run:
- logs per-epoch metrics to W&B (loss, sensitivity, specificity, AUC, confusion matrix)
- keeps the best checkpoint (by val AUC) and saves it to `models/{name}.pt`
- uploads that checkpoint as a W&B Artifact `model-{name}:vN`
- evaluates the best checkpoint on the test set and reports final metrics

### Disable W&B for a single run

```bash
uv run python -m brain_tumor_mlops.training.train model=simple_cnn no_wandb=true
```

### Tests, lint, format

```bash
uv run pytest                                # all tests
uv run pytest tests/test_splits.py -v        # one specific file
uv run ruff check .                          # lint
uv run ruff format .                         # format
```

### Regenerate the dataset (after a raw-data change)

```bash
uv run python -m brain_tumor_mlops.data.prepare
```

Rebuilds `data/processed/slice_index.parquet` + `norm_stats.json` and uploads a new version of the `lgg-mri-prepared` artifact.

Prefer `uv run dvc repro` over the raw command — it re-runs the stage **only if** raw data or `prepare.py` / `splits.py` actually changed, and updates `dvc.lock` so the new output hashes are versioned alongside the code.

### Data versioning (DVC)

Raw payload stays out of Git: only the lightweight `data/raw/kaggle_3m.dvc` pointer file is committed (~120 B). The real ~1 GB dataset lives in the DVC cache, mirrored to the configured remote.

```bash
uv run dvc status            # what is out of sync between code, cache, and remote
uv run dvc pull              # fetch raw + processed data from the remote
uv run dvc repro             # re-run the prepare_data stage if its inputs changed
uv run dvc push              # upload new versions of the data to the remote
```

**Remote backend: Google Drive.** The remote `gdrive` is declared in `.dvc/config` but its URL still contains the placeholder `gdrive://REPLACE_WITH_FOLDER_ID` until one teammate finalises the shared folder.

**First-time setup (do this once, by one person):**

1. Create a folder on Google Drive — e.g. `brain-tumor-mlops-dvc`. Share it with all 4 teammates with **Editor** access.
2. Copy the folder ID from the URL `https://drive.google.com/drive/folders/<FOLDER_ID>`.
3. Update the remote and commit the change:
   ```bash
   uv run dvc remote modify gdrive url gdrive://<FOLDER_ID>
   git add .dvc/config && git commit -m "chore(dvc): wire up shared Google Drive remote"
   ```
4. Push the cache to GDrive — the first push opens an OAuth flow in the browser:
   ```bash
   uv run dvc push
   ```
   The OAuth token is cached in `~/.cache/pydrive2fs/` (per-user, never committed).

**For every other teammate (after pulling the commit above):**

```bash
uv run dvc pull    # browser opens once for OAuth, then fetches data
```

**Alternative backends** (only relevant if we move off Drive later — e.g. quota issues, CI flakiness):

```bash
# S3 / MinIO
uv add --dev "dvc[s3]"
uv run dvc remote modify gdrive url s3://<bucket>/<path>
# then set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (+ AWS_ENDPOINT_URL for MinIO) in .env

# SSH on a shared BFH machine
uv add --dev "dvc[ssh]"
uv run dvc remote modify gdrive url ssh://user@host/path/to/dvcstore
```

See `.env.example` for the credential layout of the alternatives.

---

## Project structure

```
BrainTumor_MLOps/
├── .github/workflows/        # CI/CD
├── configs/                  # Hydra (model, training, data)
│   ├── config.yaml
│   ├── data/default.yaml
│   ├── training/default.yaml
│   └── model/{baseline,simple_cnn,unet_classifier,resnet50_transfer}.yaml
├── data/                     # gitignored, managed by DVC
│   ├── raw/                  # original TIFFs
│   └── processed/            # slice_index.parquet, norm_stats.json
├── docs/
│   └── PIPELINE.md           # full technical documentation
├── models/                   # gitignored, *.pt pulled from W&B Artifacts
├── notebooks/
│   └── 01_eda.ipynb          # exploratory data analysis (20 visualisations)
├── src/brain_tumor_mlops/
│   ├── data/                 # splits, Dataset, transforms, prep
│   ├── models/               # 4 architectures + factory
│   ├── training/             # train loop, metrics
│   ├── api/                  # FastAPI (Phase 3, upcoming)
│   ├── inference/            # prediction logic (Phase 3)
│   ├── monitoring/           # drift, metrics (Phase 5)
│   └── utils/                # wandb logging, helpers
├── tests/                    # pytest (20 tests)
├── pyproject.toml
├── dvc.yaml
├── .env.example
└── README.md
```

**Rule**: all production code lives in `src/brain_tumor_mlops/`. Notebooks are for exploration only — never import from a notebook into production code.

---

## Git workflow

### Branches

- `main` — protected, production-ready, deployable at any time.
- `dev` — integration branch.
- `feature/<short-description>` — new features.
- `fix/<short-description>` — bug fixes.
- `docs/<short-description>` — documentation only.

### Pull Requests

- PR against `dev`.
- ≥ 1 review from another team member.
- All CI checks must pass.
- **Squash merge** to keep history clean.
- Tasks tracked on the **GitHub Project** linked to the repo.

### Conventional Commits

Format: `type(scope): description`. Examples:

```
feat(models): add ResNet50 transfer learning architecture
fix(api): handle empty image upload with 422 response
docs: add monitoring setup to README
test(data): add tests for patient-level train/val split
chore(deps): bump pytorch to 2.4.0
```

---

## Useful links

- **Dataset**: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
- **W&B project**: https://wandb.ai/nathan2massicot-berner-fachhochschule/brain-tumor-classification
- **GitHub Project (task board)**: https://github.com/users/Nathan-massicot/projects/2
- **Technical documentation**: [`docs/PIPELINE.md`](docs/PIPELINE.md)

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `WANDB_API_KEY not set` | You haven't configured `.env`. Redo step 2. |
| `Could not find project brain-tumor-classification` | `WANDB_ENTITY` is missing or wrong in `.env`. Set it to `nathan2massicot-berner-fachhochschule`. |
| `brain_tumor_mlops` imports fail | You haven't run `uv sync`. The package is installed in editable mode from `pyproject.toml`. |
| First epoch endless on Mac (Apple Silicon) | Normal: MPS compiles its Metal kernel cache on the first batch (15–25 min). Subsequent epochs: ~25 s. **Don't ctrl-C.** |
| Data tests fail with `slice_index.parquet missing` | Run `uv run python -m brain_tumor_mlops.data.prepare` once. |
| `models/{name}.pt missing` or shows up as a tiny text file (~130 bytes) | You cloned without Git LFS installed. Run `brew install git-lfs && git lfs install && git lfs pull` from inside the repo. |

For anything else, ping the team on Discord/Slack, or open a GitHub issue.

## Development Setup

After cloning the repo, run the following to install hooks:

```bash
uv sync
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

Hooks run automatically on every commit:
- `ruff` — lint and auto-fix
- `ruff-format` — formatting
- `detect-private-key` — prevents accidental secret commits
- `check-merge-conflict` — catches unresolved merge conflicts
- `end-of-file-fixer` — ensures files end with a newline
- `conventional-pre-commit` — enforces conventional commit messages