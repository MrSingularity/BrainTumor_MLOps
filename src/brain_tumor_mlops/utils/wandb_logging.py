"""W&B logging helpers with graceful no-op when not configured.

Local development should never require a W&B account. Functions here detect
whether `WANDB_API_KEY` is present (or `WANDB_MODE=disabled` is set) and skip
all calls in that case, so the rest of the codebase stays free of conditional
guards.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from dotenv import load_dotenv

# Load .env from the project root so any entrypoint (scripts, tests, notebooks)
# picks up WANDB_API_KEY without needing an explicit `source .env`.
load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def wandb_enabled() -> bool:
    """Return True if W&B is configured for this process."""
    if os.getenv("WANDB_MODE") == "disabled":
        return False
    return bool(os.getenv("WANDB_API_KEY"))


@contextmanager
def wandb_run(
    *,
    project: str | None = None,
    job_type: str,
    name: str | None = None,
    config: dict | None = None,
) -> Iterator[object | None]:
    """Open a W&B run if credentials exist; yield None otherwise.

    Usage:
        with wandb_run(job_type="data-prep") as run:
            log_artifact("dataset-index", paths, run=run)
    """
    if not wandb_enabled():
        print(f"[wandb] disabled (no WANDB_API_KEY); skipping run '{job_type}'")
        yield None
        return

    import wandb

    project = project or os.getenv("WANDB_PROJECT", "brain-tumor-mlops")
    run = wandb.init(project=project, job_type=job_type, name=name, config=config or {})
    try:
        yield run
    finally:
        wandb.finish()


def log_artifact(
    name: str,
    paths: Iterable[Path | str],
    *,
    artifact_type: str = "dataset",
    description: str = "",
    run: object | None = None,
) -> None:
    """Upload one or more files as a W&B Artifact, no-op if W&B is off."""
    if run is None:
        return

    import wandb

    artifact = wandb.Artifact(name=name, type=artifact_type, description=description)
    for p in paths:
        artifact.add_file(str(p))
    run.log_artifact(artifact)
