# Hyperparameter sweep — procedure

Covers GitHub issues **#26** (sweep) and **#27** (best-model promotion).

All training runs on the **CUDA workstation** (referred to as `<remote>` below).
The Mac is only used to *create* the sweep and *inspect* the W&B dashboard.

---

## What gets swept

Defined in `configs/sweeps/resnet_unet.yaml`.

| Parameter | Search space | Distribution |
|---|---|---|
| `model` | `resnet50_transfer`, `unet_classifier` | categorical |
| `training.lr` | 1e-5 → 1e-2 | log-uniform |
| `training.weight_decay` | 1e-6 → 1e-2 | log-uniform |
| `data.batch_size` | 16, 32, 64 | discrete |
| `training.epochs` | 15 | fixed |

**Optimiser**: Bayesian. **Metric**: `val/sensitivity`, maximised (missed
tumours are the worst error in a medical context). **Early termination**:
Hyperband kills unpromising runs after 3 epochs to save compute.

---

## One-shot launch

### 1. Create the sweep (Mac, once)

```bash
uv run wandb sweep configs/sweeps/resnet_unet.yaml
# → Sweep created.
# → wandb: Sweep ID: abc1234
# → wandb: View sweep at: https://wandb.ai/MLopsTeamMsC/brain-tumor-classification/sweeps/abc1234
# → wandb: Run sweep agent with: wandb agent MLopsTeamMsC/brain-tumor-classification/abc1234
```

Copy the **sweep ID** — you'll need it on the remote.

### 2. Prepare the CUDA workstation (remote, once)

```bash
ssh <remote>

# Clone + LFS (model checkpoints are LFS-tracked)
git clone https://github.com/Nathan-massicot/BrainTumor_MLOps.git
cd BrainTumor_MLOps
sudo apt-get install -y git-lfs        # if missing
git lfs install && git lfs pull

# uv toolchain + deps
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Secrets
cp .env.example .env                   # then edit: WANDB_API_KEY=... WANDB_ENTITY=MLopsTeamMsC

# Data — needs the DVC Google Drive folder ID to be wired in .dvc/config first
uv run dvc pull
```

Sanity-check CUDA before starting the agent:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → True NVIDIA ...
```

The `device: auto` default in `configs/config.yaml` picks CUDA when available,
so no flag needs to be passed.

### 3. Start the agent (remote, in tmux)

```bash
tmux new -s sweep

uv run wandb agent MLopsTeamMsC/brain-tumor-classification/<sweep-id> --count 20
# Ctrl-B then D to detach. `tmux attach -t sweep` to come back.
```

- `--count 20` caps this agent at 20 runs. Drop the flag for unbounded.
- The agent fetches one config at a time from the W&B server and runs the
  training command from the sweep YAML — no further wiring needed.

### 4. (Optional) Parallel agents

Run the same `wandb agent ...` command in a second tmux window or on another
GPU machine. W&B distributes configs automatically; all runs land in the same
sweep.

---

## Monitor

- **Dashboard**: `https://wandb.ai/MLopsTeamMsC/brain-tumor-classification/sweeps/<sweep-id>`
- **CLI** (any machine with `WANDB_API_KEY`):
  ```bash
  uv run wandb sweep --status MLopsTeamMsC/brain-tumor-classification/<sweep-id>
  ```

Per-run logs land under `outputs/<date>/<time>_<model>/` on the remote (gitignored).

---

## Pick the winner & promote (closes #27)

Once the sweep completes:

1. On the dashboard, sort runs by `val/sensitivity` (or use the "Best run" panel).
2. Click the winning run → the **Artifacts** tab shows `model-<arch>:v<N>`.
3. Add the alias `champion` to that artifact version (UI: Aliases → "+ Add").
4. (Optional) Link the artifact to the **Model Registry** entry
   `brain-tumor-detector` with the alias `production-candidate`.

**The link to the Model Registry is the mandatory human gate** for medical
model promotion (see `CLAUDE.md` — "Production model promotion requires human
validation"). Never automate this step.

To use the champion locally after promotion:

```bash
uv run wandb artifact get \
  MLopsTeamMsC/brain-tumor-classification/model-resnet50_transfer:champion \
  --root models/
```

---

## Stop the sweep early

If you've seen enough:

```bash
uv run wandb sweep --cancel MLopsTeamMsC/brain-tumor-classification/<sweep-id>
```

Running agents will finish their current run and then exit cleanly.
