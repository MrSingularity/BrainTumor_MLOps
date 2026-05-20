"""End-to-end train + test loop, driven by Hydra.

Usage:
    # Train the default model (simple_cnn) with default settings:
    uv run python -m mlops_project.training.train

    # Override one or more values from the CLI (Hydra syntax):
    uv run python -m mlops_project.training.train model=resnet50_transfer
    uv run python -m mlops_project.training.train model=unet_classifier training.epochs=10
    uv run python -m mlops_project.training.train data.batch_size=64 training.lr=1e-4

    # Run a multi-run sweep over models in one command:
    uv run python -m mlops_project.training.train --multirun model=baseline,simple_cnn,unet_classifier,resnet50_transfer

The script:
    1. Loads the prepared slice index + normalisation stats (`prepare.py` outputs).
    2. Builds train/val/test DataLoaders (train uses augmentations).
    3. Runs `training.epochs` rounds, logging per-epoch medical metrics
       (sensitivity, specificity, AUC) to W&B if WANDB_API_KEY is set.
    4. Saves the best-val-AUC checkpoint and uploads it as a W&B Artifact
       tagged with the model name.
    5. Evaluates the best checkpoint on the held-out test split.

Picks MPS on Apple Silicon when available, then CUDA, else CPU.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from mlops_project.data.dataset import BrainMRIDataset, load_dataset_artifacts
from mlops_project.data.transforms import eval_transform, train_transform
from mlops_project.models.factory import build_model, count_parameters
from mlops_project.training.metrics import classification_metrics
from mlops_project.utils.wandb_logging import log_artifact, wandb_run

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(spec)


def _make_loaders(processed_dir: Path, batch_size: int, num_workers: int) -> dict[str, DataLoader]:
    index, stats = load_dataset_artifacts(processed_dir)
    common = dict(index=index, stats=stats)
    train_ds = BrainMRIDataset(**common, split="train", transform=train_transform())
    val_ds = BrainMRIDataset(**common, split="val", transform=eval_transform())
    test_ds = BrainMRIDataset(**common, split="test", transform=eval_transform())
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers),
    }


def _pos_weight_from_train(loader: DataLoader) -> float:
    pos = neg = 0
    for batch in loader:
        labels = batch["label"]
        pos += int(labels.sum().item())
        neg += int((labels == 0).sum().item())
    return max(neg / max(pos, 1), 1.0)


def _run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimiser: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimiser is not None
    model.train(is_train)
    total_loss, n = 0.0, 0
    y_true_chunks, y_prob_chunks = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            if is_train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total_loss += float(loss.item()) * x.size(0)
            n += x.size(0)
            y_true_chunks.append(y.detach().cpu().numpy())
            y_prob_chunks.append(torch.sigmoid(logits).detach().cpu().numpy())

    return (
        total_loss / max(n, 1),
        np.concatenate(y_true_chunks),
        np.concatenate(y_prob_chunks),
    )


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    project_root = Path(cfg.paths.project_root).resolve()
    processed_dir = Path(cfg.paths.processed)
    models_dir = Path(cfg.paths.models)
    models_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg.device)
    print(f"[setup] device={device}  config={OmegaConf.to_yaml(cfg.model)}".replace("\n", " "))

    loaders = _make_loaders(processed_dir, cfg.data.batch_size, cfg.data.num_workers)
    pos_weight = _pos_weight_from_train(loaders["train"])
    print(f"[setup] pos_weight={pos_weight:.3f}  "
          f"batches train={len(loaders['train'])} val={len(loaders['val'])} test={len(loaders['test'])}")

    model = build_model(cfg.model.name, **dict(cfg.model.kwargs)).to(device)
    total, trainable = count_parameters(model)
    print(f"[setup] model={cfg.model.name}  params total={total:,} trainable={trainable:,}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    flat_config = {
        "model": cfg.model.name,
        "epochs": cfg.training.epochs,
        "batch_size": cfg.data.batch_size,
        "lr": cfg.training.lr,
        "weight_decay": cfg.training.weight_decay,
        "pos_weight": pos_weight,
        "params_total": total,
        "params_trainable": trainable,
        "device": str(device),
        "seed": cfg.seed,
    }

    best_state, best_val_auc, history = None, -1.0, []

    with wandb_run(
        job_type="train",
        name=f"{cfg.model.name}-e{cfg.training.epochs}",
        config=flat_config,
    ) as run:
        for epoch in range(1, cfg.training.epochs + 1):
            t0 = time.time()
            train_loss, y_t_train, y_p_train = _run_one_epoch(
                model, loaders["train"], optimiser=optim, loss_fn=loss_fn, device=device
            )
            val_loss, y_t_val, y_p_val = _run_one_epoch(
                model, loaders["val"], optimiser=None, loss_fn=loss_fn, device=device
            )
            train_m = classification_metrics(y_t_train, y_p_train)
            val_m = classification_metrics(y_t_val, y_p_val)
            took = time.time() - t0

            row = {
                "epoch": epoch,
                "time_s": round(took, 1),
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"train/{k}": v for k, v in train_m.as_dict().items()},
                **{f"val/{k}": v for k, v in val_m.as_dict().items()},
            }
            history.append(row)
            print(
                f"[epoch {epoch:>2}/{cfg.training.epochs}] "
                f"loss train={train_loss:.4f} val={val_loss:.4f} | "
                f"val {val_m.pretty()}  ({took:.1f}s)"
            )
            if run is not None:
                run.log(row, step=epoch)

            if val_m.auc_roc > best_val_auc:
                best_val_auc = val_m.auc_roc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        _, y_t_test, y_p_test = _run_one_epoch(
            model, loaders["test"], optimiser=None, loss_fn=loss_fn, device=device
        )
        test_m = classification_metrics(y_t_test, y_p_test)
        print(f"[TEST] {test_m.pretty()}")

        ckpt_path = models_dir / f"{cfg.model.name}.pt"
        torch.save(
            {
                "model_name": cfg.model.name,
                "state_dict": best_state if best_state is not None else model.state_dict(),
                "config": flat_config,
                "best_val_auc": best_val_auc,
                "test_metrics": test_m.as_dict(),
                "history": history,
                "hydra_cfg": OmegaConf.to_container(cfg, resolve=True),
            },
            ckpt_path,
        )
        results_path = models_dir / f"{cfg.model.name}_results.json"
        results_path.write_text(json.dumps({
            "model": cfg.model.name,
            "config": flat_config,
            "best_val_auc": best_val_auc,
            "test": test_m.as_dict(),
            "history": history,
        }, indent=2))
        print(f"[save] checkpoint → {ckpt_path}")
        print(f"[save] results   → {results_path}")

        if run is not None:
            run.summary.update({f"test/{k}": v for k, v in test_m.as_dict().items()})
            run.summary["best_val_auc"] = best_val_auc
            log_artifact(
                f"model-{cfg.model.name}",
                paths=[ckpt_path, results_path],
                artifact_type="model",
                description=(
                    f"{cfg.model.name} trained {cfg.training.epochs} epochs, "
                    f"best val AUC={best_val_auc:.3f}, test AUC={test_m.auc_roc:.3f}."
                ),
                run=run,
            )


if __name__ == "__main__":
    main()
