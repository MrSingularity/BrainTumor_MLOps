# Brain Tumor MLOps — Detection & Localization

End-to-end MLOps pipeline for **brain tumor detection and localization** from MRI images. The project covers the full ML lifecycle: data ingestion, training, inference API, frontend, monitoring, drift detection, and continuous retraining.

> ⚠️ **Disclaimer**: this is an **academic / educational project**. It is **not intended for clinical use**. No medical validation, no CE/FDA certification.



## Context

- **Course**: MSc-level MLOps module — graded project requiring at least 3 MLOps tools.
- **Team**: Gabriel Gillmann · Helena Martínez Río · Nathan Massicot · Jahnavi Patil.
- **Dataset**: [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (Kaggle).
- **Task**: binary classification (tumor / no tumor) on MRI slices, **plus pixel-level localization** with a bounding box overlay on the scan.
- **Models**: 4 classification architectures (`baseline`, `simple_cnn`, `unet_classifier`, `resnet50_transfer`) + 2 segmentation architectures (`mini_unet`, `unet_segmentation`) for localization.

📘 **Full technical documentation**: [`docs/PIPELINE.md`](docs/PIPELINE.md) — data pipeline, models, results, architecture choices.
---


## Prerequisites

- **Python 3.12**
- **[uv](https://docs.astral.sh/uv/)** — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Git** + **Git LFS** — `brew install git-lfs && git lfs install` (macOS) — required to fetch the model checkpoints in `models/*.pt`
- A free **Weights & Biases** account — https://wandb.ai
- A **Kaggle** account — to download the MRI dataset (~140 MB)
- **Docker** (optional, for the full local stack)
- **GPU** recommended for training (CUDA / MPS); inference runs on CPU

---

##  Launch the App 


### 1. Install dependencies

```bash
git clone https://github.com/Nathan-massicot/BrainTumor_MLOps.git
cd BrainTumor_MLOps
git lfs install && git lfs pull          # pulls model checkpoints (LFS)
uv sync --all-extras                     # creates .venv + installs everything
```

### 2. Get a Kaggle API token (for the dataset)

The MRI dataset isn't shipped with the repo — you download it directly from Kaggle:

1. Create a free Kaggle account at https://www.kaggle.com
2. Go to **Account → API → Create New Token** — downloads a `kaggle.json` file
3. Put your credentials in `.env`:
   ```bash
   cp .env.example .env
   ```
   Then set `KAGGLE_USERNAME` and `KAGGLE_KEY` (from the downloaded `kaggle.json`).

> **Optional for the launch**: `WANDB_API_KEY` and `WANDB_ENTITY=nathan2massicot-berner-fachhochschule` — only needed if you intend to **train** new models. The frontend and inference work without W&B.

### 3. Download the dataset + prepare it

```bash
# Download the raw MRI dataset (~140 MB) from Kaggle
uv run kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p data/raw --unzip
mv data/raw/lgg-mri-segmentation data/raw/kaggle_3m

# Build the slice index + per-channel normalization stats used by the frontend
uv run python -m brain_tumor_mlops.data.prepare
```

📂 Dataset page: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

All 6 `.pt` model checkpoints — including the **`resnet50_transfer.pt`** champion (94 MB) — come from **Git LFS** at clone time. Nothing else to download.

### 4. Launch the Streamlit frontend

```bash
uv run streamlit run frontend/app.py
```

Opens automatically at **http://localhost:8501**. From the sidebar:
- Pick a **classification checkpoint** (default: `baseline.pt` — switch to `resnet50_transfer.pt` for best results)
- Enable **"Localize tumor when detected"** → pick a segmentation model (`unet_segmentation.pt` is the most accurate)
- Click any sample image (or upload your own MRI) → the app runs classification, and if tumor is detected, overlays a **matte-gold bounding box** on the scan from the segmentation model.

### 5. (Optional) Run the full Docker stack

For the production-like setup with API + frontend + Prometheus + Grafana:

```bash
cp .env.example .env                     # only needed once
docker compose up -d
```

- Frontend: http://localhost:8501
- API docs: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin / admin)


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




## Daily workflow

### Run a training

**Classification** (tumor / no tumor — 4 architectures):

```bash
# Default model (simple_cnn, 5 epochs, batch 32)
uv run python -m brain_tumor_mlops.training.train

# Pick one of the 4 architectures
uv run python -m brain_tumor_mlops.training.train model=resnet50_transfer
uv run python -m brain_tumor_mlops.training.train model=unet_classifier

# Override hyperparameters from the CLI (Hydra syntax)
uv run python -m brain_tumor_mlops.training.train model=resnet50_transfer training.epochs=20 training.lr=1e-4 data.batch_size=64

# Train all 4 classification models in a row (multirun)
uv run python -m brain_tumor_mlops.training.train --multirun \
    model=baseline,simple_cnn,unet_classifier,resnet50_transfer training.epochs=10
```

**Segmentation** (tumor localization — 2 architectures, BCE + Dice loss):

```bash
# Mini U-Net (~117K params, ~30s/epoch on MPS) — fast baseline
uv run python -m brain_tumor_mlops.training.train_segmentation \
    --config-name=config_segmentation model=mini_unet

# Full U-Net (~7.7M params, ~75s/epoch on MPS) — best Dice
uv run python -m brain_tumor_mlops.training.train_segmentation \
    --config-name=config_segmentation model=unet_segmentation training.epochs=20
```

Both write to `models/{name}.pt` with the same checkpoint layout as classification (so `load_checkpoint` works the same), plus a `test_metrics` block containing `dice`, `iou`, `pixel_accuracy`.

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

Raw payload stays out of Git: only the lightweight `data/raw/kaggle_3m.dvc` pointer file is committed (~120 B). The real ~1 GB dataset lives in the DVC cache, mirrored to **Backblaze B2** (S3-compatible API).

```bash
uv run dvc status            # what is out of sync between code, cache, and remote
uv run dvc pull              # fetch raw + processed data + champion model from B2
uv run dvc repro             # re-run the prepare_data stage if its inputs changed
uv run dvc push              # upload new versions of the data/models to B2
```

**Remote backend: Backblaze B2** (S3-compatible). Declared in `.dvc/config`:

```ini
['remote "b2"']
    url = s3://brain-tumor-mlops-nm/dvcstore
    endpointurl = https://s3.eu-central-003.backblazeb2.com
```

**Credentials** go in `.dvc/config.local` (gitignored). The team's keys are in the shared password vault; ask a teammate. Format:

```ini
['remote "b2"']
    access_key_id = <your-b2-application-key-id>
    secret_access_key = <your-b2-application-key>
```

If you generate your own B2 key, make sure to enable **"Allow List All Bucket Names"** so `HeadObject` works for DVC's existence checks.

**What's tracked by DVC vs Git LFS:**

| Artifact | Where | Why |
|---|---|---|
| `data/raw/kaggle_3m/` (~1 GB, 7860 files) | DVC → B2 | Way too big for Git/LFS |
| `data/processed/*` (~5 MB) | DVC → B2 | Versioned alongside raw data |
| `models/resnet50_transfer.pt` (94 MB) | DVC → B2 | The production champion |
| `models/{baseline,simple_cnn,unet_classifier,mini_unet,unet_segmentation}.pt` | **Git LFS** | Small enough (~5 KB → 31 MB), faster for new joiners |

> ⚠️ B2 free tier has a **2 500 Class B transactions/day** cap. A full `dvc pull` of the dataset uses ~3 000 ops → likely to hit the cap. If you see `403 Forbidden` on `HeadObject`, check https://secure.backblaze.com/account.htm → Caps & Alerts.

---

## Project structure

```
BrainTumor_MLOps/
├── .github/workflows/        # CI/CD (test + docker-smoke)
├── configs/                  # Hydra configs
│   ├── config.yaml           # classification root config
│   ├── config_segmentation.yaml  # segmentation root config
│   ├── data/default.yaml
│   ├── training/{default,segmentation}.yaml
│   └── model/{baseline,simple_cnn,unet_classifier,resnet50_transfer,
│              unet_segmentation,mini_unet}.yaml
├── data/                     # gitignored, raw + processed via DVC
│   ├── raw/kaggle_3m/        # original TIFFs (DVC → B2)
│   └── processed/            # slice_index.parquet, norm_stats.json (DVC)
├── docker/                   # Dockerfile per service
├── docs/PIPELINE.md          # full technical documentation
├── frontend/
│   ├── app.py                # NeuroScan Streamlit UI (classification + localization)
│   └── ops_dashboard.py      # ops/health dashboard
├── models/                   # *.pt — 5 via Git LFS + resnet50_transfer via DVC
├── monitoring/grafana/       # Grafana dashboards + alert rules
├── notebooks/01_eda.ipynb    # exploratory data analysis
├── src/brain_tumor_mlops/
│   ├── api/                  # FastAPI app + metrics + logs viewer
│   ├── data/                 # splits, Dataset, transforms, prep
│   ├── models/               # 6 architectures + factory
│   ├── monitoring/           # drift job + metrics
│   ├── training/             # train + train_segmentation + metrics
│   └── utils/                # wandb logging, helpers
├── tests/                    # pytest (200+ tests)
├── docker-compose.yml
├── pyproject.toml
├── dvc.yaml
├── .gitattributes            # Git LFS rules for models/*.pt
├── .env.example
└── README.md
```

**Rule**: all production code lives in `src/brain_tumor_mlops/`. Notebooks are for exploration only — never import from a notebook into production code.


## Useful links

- **Dataset**: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
- **W&B project**: https://wandb.ai/nathan2massicot-berner-fachhochschule/brain-tumor-classification
- **GitHub Project (task board)**: https://github.com/users/Nathan-massicot/projects/2
- **Technical documentation**: [`docs/PIPELINE.md`](docs/PIPELINE.md)

