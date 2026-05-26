"""Hydra-driven training loop for binary tumour segmentation.

Mirrors training/train.py but with:
    * a BCE + Dice combined loss,
    * Dice / IoU / pixel-accuracy metrics,
    * checkpoints that select on best val Dice.

Usage:
    uv run python -m brain_tumor_mlops.training.train_segmentation \
        --config-name=config_segmentation model=unet_segmentation
    uv run python -m brain_tumor_mlops.training.train_segmentation \
        --config-name=config_segmentation model=mini_unet training.epochs=20
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

from brain_tumor_mlops.data.dataset import BrainMRIDataset, load_dataset_artifacts
from brain_tumor_mlops.data.transforms import eval_transform, train_transform
from brain_tumor_mlops.models.factory import build_model, count_parameters
from brain_tumor_mlops.training.metrics import segmentation_metrics
from brain_tumor_mlops.utils.wandb_logging import log_artifact, wandb_run


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
    common = dict(index=index, stats=stats, return_mask=True)
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


class BCEDiceLoss(nn.Module):
    """BCE + soft-Dice. Robust default for binary segmentation under imbalance."""

    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        inter = (probs * target).sum(dim=(1, 2, 3))
        sums = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * inter + 1.0) / (sums + 1.0))
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice.mean()


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
            y = batch["mask"].to(device, non_blocking=True)
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


@hydra.main(version_base=None, config_path="../../../configs", config_name="config_segmentation")
def main(cfg: DictConfig) -> None:
    if cfg.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    processed_dir = Path(cfg.paths.processed)
    models_dir = Path(cfg.paths.models)
    models_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg.device)
    print(f"[setup] device={device}  model={cfg.model.name}")

    loaders = _make_loaders(processed_dir, cfg.data.batch_size, cfg.data.num_workers)
    print(
        f"[setup] batches train={len(loaders['train'])} "
        f"val={len(loaders['val'])} test={len(loaders['test'])}"
    )

    model = build_model(cfg.model.name, **dict(cfg.model.kwargs)).to(device)
    total, trainable = count_parameters(model)
    print(f"[setup] params total={total:,} trainable={trainable:,}")

    loss_fn = BCEDiceLoss(bce_weight=cfg.training.bce_weight)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    flat_config = {
        "model": cfg.model.name,
        "task": "segmentation",
        "epochs": cfg.training.epochs,
        "batch_size": cfg.data.batch_size,
        "lr": cfg.training.lr,
        "weight_decay": cfg.training.weight_decay,
        "bce_weight": cfg.training.bce_weight,
        "params_total": total,
        "params_trainable": trainable,
        "device": str(device),
        "seed": cfg.seed,
    }

    best_state, best_val_dice, history = None, -1.0, []

    with wandb_run(
        job_type="train-segmentation",
        name=f"{cfg.model.name}-seg-e{cfg.training.epochs}",
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
            train_m = segmentation_metrics(y_t_train, y_p_train)
            val_m = segmentation_metrics(y_t_val, y_p_val)
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

            if val_m.dice > best_val_dice:
                best_val_dice = val_m.dice
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        _, y_t_test, y_p_test = _run_one_epoch(
            model, loaders["test"], optimiser=None, loss_fn=loss_fn, device=device
        )
        test_m = segmentation_metrics(y_t_test, y_p_test)
        print(f"[TEST] {test_m.pretty()}")

        ckpt_path = models_dir / f"{cfg.model.name}.pt"
        torch.save(
            {
                "model_name": cfg.model.name,
                "task": "segmentation",
                "state_dict": best_state if best_state is not None else model.state_dict(),
                "config": flat_config,
                "best_val_dice": best_val_dice,
                "test_metrics": test_m.as_dict(),
                "history": history,
                "hydra_cfg": OmegaConf.to_container(cfg, resolve=True),
            },
            ckpt_path,
        )
        results_path = models_dir / f"{cfg.model.name}_results.json"
        results_path.write_text(json.dumps({
            "model": cfg.model.name,
            "task": "segmentation",
            "config": flat_config,
            "best_val_dice": best_val_dice,
            "test": test_m.as_dict(),
            "history": history,
        }, indent=2))
        print(f"[save] checkpoint → {ckpt_path}")
        print(f"[save] results   → {results_path}")

        if run is not None:
            run.summary.update({f"test/{k}": v for k, v in test_m.as_dict().items()})
            run.summary["best_val_dice"] = best_val_dice
            log_artifact(
                f"model-{cfg.model.name}",
                paths=[ckpt_path, results_path],
                artifact_type="model",
                description=(
                    f"{cfg.model.name} (segmentation) trained {cfg.training.epochs} epochs, "
                    f"best val Dice={best_val_dice:.3f}, test Dice={test_m.dice:.3f}."
                ),
                run=run,
            )


if __name__ == "__main__":
    main()
