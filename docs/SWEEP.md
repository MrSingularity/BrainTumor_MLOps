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

> The remote is a **Windows machine** reached via OpenSSH. Pick **one** of the
> two paths below. WSL2 is strongly recommended: NVIDIA's CUDA-on-WSL has been
> stable since 2021, and all the existing tooling (`uv`, `tmux`, `dvc`) just
> works as on Linux.

#### Path A — WSL2 (recommended)

Prereqs on the Windows host (one-time, admin):
1. `wsl --install -d Ubuntu` (PowerShell, admin) — installs WSL2 + Ubuntu.
2. NVIDIA driver ≥ R535 on Windows. **Do NOT install a CUDA toolkit inside WSL** —
   CUDA libs come from the Windows driver via `/usr/lib/wsl/lib`.

Then, from the Mac:
```bash
ssh <user>@<windows-host>            # lands in PowerShell
wsl                                   # drop into the Ubuntu shell
```

Inside WSL Ubuntu, run the standard Linux setup:
```bash
sudo apt-get update && sudo apt-get install -y git git-lfs curl
git clone https://github.com/Nathan-massicot/BrainTumor_MLOps.git
cd BrainTumor_MLOps
git lfs install && git lfs pull

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync

cp .env.example .env                  # then edit: WANDB_API_KEY=... WANDB_ENTITY=MLopsTeamMsC
uv run dvc pull                       # needs the GDrive folder ID wired in .dvc/config
```

Sanity-check CUDA:
```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → True NVIDIA GeForce RTX ...
```

#### Path B — Native Windows + PowerShell

Open a PowerShell session via SSH and run:
```powershell
# Tooling (winget ships with Windows 11)
winget install --id Git.Git -e
winget install --id GitHub.GitLFS -e
winget install --id astral-sh.uv -e   # uv has a native Windows installer
# Refresh PATH in the current session
$env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")

# Repo
git clone https://github.com/Nathan-massicot/BrainTumor_MLOps.git
cd BrainTumor_MLOps
git lfs install
git lfs pull

uv sync

# Secrets
Copy-Item .env.example .env           # then edit with notepad .env (set WANDB_API_KEY, WANDB_ENTITY)
uv run dvc pull
```

Sanity-check CUDA: same `uv run python -c "..."` command.

`configs/config.yaml`'s `device: auto` picks CUDA on either path — nothing else
to configure.

### 3. Start the agent (remote, must survive SSH disconnect)

#### Path A — WSL2 (tmux works)
```bash
tmux new -s sweep
uv run wandb agent MLopsTeamMsC/brain-tumor-classification/<sweep-id> --count 20
# Ctrl-B then D to detach. `tmux attach -t sweep` to reattach.
```

#### Path B — Native Windows (no tmux; use `Start-Process`)
```powershell
$out = "$PWD\sweep.log"
Start-Process -FilePath "uv" `
  -ArgumentList "run","wandb","agent","MLopsTeamMsC/brain-tumor-classification/<sweep-id>","--count","20" `
  -RedirectStandardOutput $out `
  -RedirectStandardError "$PWD\sweep.err" `
  -WindowStyle Hidden
# The process now owns its own PID and survives SSH disconnect.
# Tail the log live (from another SSH session) with:  Get-Content -Wait .\sweep.log
```

To kill it later:
```powershell
Get-Process | Where-Object { $_.MainModule.FileName -like "*uv*" } | Stop-Process
```

In both cases:
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
