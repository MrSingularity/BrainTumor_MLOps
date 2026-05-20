# CLAUDE.md

This file provides guidance to Claude (and other LLM assistants) when working with code in this repository.

## Project Overview

**Brain Tumor Detection & Localization** — A complete MLOps project that classifies brain MRI images as tumor / no-tumor using a CNN with transfer learning. The project is built end-to-end: data pipeline, model training, inference API, frontend, monitoring, drift detection, and continuous retraining.

**Course context:** MSc-level MLOps module project (≥3 MLOps tools required, graded).

**Team:** Gabriel Gillmann, Helena Martínez Río, Nathan Massicot, Jahnavi Patil.

**Dataset:** [LGG MRI Segmentation (Kaggle)](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Package & env management | **uv** (Astral) |
| ML framework | **PyTorch** + torchvision (transfer learning) |
| Experiment tracking | **Weights & Biases (W&B)** |
| Config management | **Hydra** / OmegaConf |
| Data versioning | **DVC** |
| API | **FastAPI** |
| Frontend | **Streamlit** (or React TBD) |
| Containerization | **Docker** + docker-compose |
| CI/CD | **GitHub Actions** |
| Monitoring | **Prometheus** + **Grafana** |
| Drift detection | **Evidently AI** |
| Orchestration / retraining | **Prefect** (or Airflow) |
| Model registry | **W&B Model Registry** (or MLflow) |
| Testing | **pytest** + httpx |
| Linting / formatting | **Ruff** |
| Pre-commit | **pre-commit** + conventional commits |

---

## Project Structure

```
brain-tumor-mlops/
├── .github/workflows/        # CI/CD pipelines
├── configs/                  # Hydra configs (model, training, data, ...)
├── data/                     # DVC-tracked, never commit raw data to Git
├── docker/                   # Dockerfiles per service
├── docs/                     # MkDocs documentation
├── notebooks/                # EDA, experiments — NOT for production code
├── monitoring/               # Grafana dashboards, Prometheus config
├── pipelines/                # Prefect flows (training, retraining, drift jobs)
├── src/brain_tumor_mlops/
│   ├── api/                  # FastAPI app, routes, schemas
│   ├── data/                 # Dataset, transforms, loaders
│   ├── models/               # CNN architectures
│   ├── training/             # Train/val loops, callbacks
│   ├── inference/            # Predict logic, model loading
│   ├── monitoring/           # Drift detection, metrics
│   └── utils/                # Logging, helpers
├── tests/                    # Unit + integration tests
├── frontend/                 # Streamlit / React UI
├── pyproject.toml
├── docker-compose.yml
├── dvc.yaml
├── .pre-commit-config.yaml
├── .env.example
└── README.md
```

**Rule:** all production code lives in `src/brain_tumor_mlops/`. Notebooks are for exploration only — never import from notebooks.

---

## Common Commands

### Setup (first time)

```bash
git clone <repo>
cd brain-tumor-mlops
uv sync                                      # install all deps incl. dev
cp .env.example .env                         # then fill in real values
pre-commit install
pre-commit install --hook-type commit-msg
dvc pull                                     # fetch tracked datasets/models
```

### Development

```bash
uv run pytest                                # run all tests
uv run pytest tests/test_models.py -v        # run a specific test file
uv run ruff check .                          # lint
uv run ruff format .                         # format
pre-commit run --all-files                   # run all hooks manually
```

### Training

```bash
uv run python -m brain_tumor_mlops.training.train          # default config
uv run python -m brain_tumor_mlops.training.train model=resnet50 training.lr=1e-4
uv run wandb sweep configs/sweeps/lr_sweep.yaml            # hyperparam sweep
```

### API & Frontend

```bash
uv run uvicorn brain_tumor_mlops.api.main:app --reload     # dev server
uv run streamlit run frontend/app.py                       # frontend
docker-compose up                                          # full stack
```

### Data & Models

```bash
dvc add data/raw/lgg-mri                     # track new dataset
dvc push                                     # push to remote storage
dvc repro                                    # rerun pipeline if inputs changed
```

### Monitoring

```bash
docker-compose up prometheus grafana          # spin up monitoring stack
uv run python -m brain_tumor_mlops.monitoring.drift_check  # manual drift report
```

---

## Code Conventions

### Python style

- **Formatter & linter:** Ruff (config in `pyproject.toml`).
- **Type hints:** required on all public functions and class methods.
- **Docstrings:** Google-style for modules, classes, and public functions.
- **Imports:** sorted by Ruff (`isort` rules).
- **Line length:** 100 chars.

### Naming

- Modules and files: `snake_case.py`
- Classes: `PascalCase`
- Functions and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: prefix with `_`

### Configuration

- **Never hardcode** paths, hyperparameters, or credentials in code.
- All configs go in `configs/` (Hydra YAML).
- All secrets go in `.env` (loaded via `python-dotenv` or `pydantic-settings`).
- `.env` is in `.gitignore`. Document required vars in `.env.example`.

### Logging

- Use the `logging` module, not `print()`.
- Configure root logger in `src/brain_tumor_mlops/utils/logging.py`.
- Log levels: DEBUG (dev), INFO (default prod), WARNING/ERROR for issues.

---

## Git Workflow

### Branches

- `main` — protected, production-ready, deployable at any time.
- `dev` — integration branch.
- `feature/<short-description>` — new features.
- `fix/<short-description>` — bug fixes.
- `docs/<short-description>` — documentation only.

### Commit messages — Conventional Commits

Format: `type(scope): description`

**Types:**
- `feat` — new feature
- `fix` — bug fix
- `docs` — documentation
- `test` — tests
- `refactor` — code change, no behavior change
- `perf` — performance improvement
- `chore` — tooling, deps, config
- `style` — formatting, whitespace

**Examples:**
```
feat(models): add ResNet50 transfer learning architecture
fix(api): handle empty image upload with 422 response
docs: add monitoring setup to README
test(data): add tests for patient-level train/val split
chore(deps): bump pytorch to 2.4.0
```

### Pull requests

- Open PR against `dev`.
- Require ≥1 review from another team member.
- All CI checks must pass.
- Squash merge to keep history clean.

---

## Critical Rules

### Data integrity

- **Patient-level splits only.** Never split by image — same patient in train and val = data leakage.
- **No raw data in Git.** Always use DVC.
- **No PHI** (protected health information) in commits, logs, or W&B.

### Model artifacts

- Models are versioned in **W&B Model Registry**, not Git.
- Production model promotion requires **human validation** (medical context — non-negotiable).
- Tag releases of the production model: `v1.0.0`, `v1.1.0`, etc.

### Secrets

- Never commit `.env`, API keys, tokens, passwords.
- `pre-commit` runs `detect-private-key` to catch accidents.
- If a secret leaks: rotate it immediately, then clean Git history.

### File size

- `pre-commit` blocks files > 500 KB.
- Large files (data, models, images) → DVC.

---

## Testing Strategy

| Layer | Tool | Goal |
|-------|------|------|
| Unit | pytest | Test individual functions (preprocessing, metrics, transforms) |
| Integration | pytest + httpx | Test API endpoints end-to-end |
| Data | pytest | Validate dataset shape, splits, no leakage |
| Model | pytest | Smoke test training loop on a tiny subset |
| E2E | manual / GH Actions | Full Docker stack up, predict, check response |

**Coverage target:** ≥70% on `src/brain_tumor_mlops/`.

---

## MLOps-Specific Guidance

### Drift detection (Phase 5)

Three drift types to monitor:

1. **Data drift** — input distribution shift. Use Evidently with KS test on pixel statistics + Wasserstein distance on CNN embeddings (penultimate layer).
2. **Concept drift** — output distribution shift (proportion of positive predictions over time).
3. **Performance drift** — accuracy degradation (requires labeled feedback from radiologists).

Drift jobs run as scheduled Prefect flows. Reports stored in S3/MinIO. Alerts go to Slack/email if threshold exceeded.

### Retraining loop (Phase 6)

Triggers (any of):
- Drift detection threshold exceeded
- Scheduled (weekly)
- Manual trigger via Prefect UI
- N new labeled samples accumulated

Pipeline: collect → preprocess → train → evaluate → compare to champion → **human review** → promote.

**Champion/Challenger:** new model runs in shadow mode for 1–2 weeks before promotion. Never auto-promote in medical contexts.

### Monitoring metrics (FastAPI → Prometheus)

Per-prediction logs:
- Inference latency (p50, p95, p99)
- Model confidence (softmax max)
- Image stats (mean, std, resolution)
- Prediction class distribution
- Request rate, error rate

Grafana dashboards: API health, model behavior, drift status.

---

## Domain Context (Medical Imaging)

- **MRI conventions:** images come as 2D slices from 3D volumes. Be careful with augmentations — horizontal flip is OK, vertical flip usually is not (anatomical correctness).
- **Class imbalance** is common. Use weighted loss or focal loss if needed.
- **Metrics matter:** raw accuracy is misleading. Always report **sensitivity (recall)**, **specificity**, **AUC-ROC**, and confusion matrix. In medical settings, false negatives (missed tumors) are far worse than false positives.
- **Regulation:** any real deployment would require CE marking (EU MDR) or FDA clearance. This project is **research/educational only** — not for clinical use. Make this explicit in the UI.

---

## When Working on This Project, Claude Should:

1. **Respect the structure** — production code in `src/`, configs in `configs/`, never bypass.
2. **Use uv** for any package addition (`uv add <pkg>` or `uv add --dev <pkg>`).
3. **Add type hints and docstrings** to any new function.
4. **Write tests alongside features** — if you add a function, add its test.
5. **Use Hydra configs** instead of hardcoding hyperparameters.
6. **Log via the configured logger**, not `print`.
7. **Follow conventional commits** when suggesting commit messages.
8. **Flag any code that could leak data** between train/val/test splits.
9. **Prioritize medical metrics** (sensitivity, specificity) over raw accuracy in suggestions.
10. **Never suggest auto-promoting models** to production without human review — this is a hard rule for the medical context.

---

## Useful Links

- Dataset: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
- W&B project: _to be added_
- Deployed app: _to be added_
- Internal docs: _to be added_