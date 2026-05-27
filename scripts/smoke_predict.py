"""Smoke-test the /predict endpoint against real MRI scans.

Sends a batch of `.tif` images from the dataset to a running API instance and
prints each prediction. Intended for quick manual verification that the API and
a checkpoint are wired up correctly — not a substitute for the pytest suite.

Examples:
    # Run the API first (uv run uvicorn brain_tumor_mlops.api.main:app)
    uv run python scripts/smoke_predict.py
    uv run python scripts/smoke_predict.py --limit 50 --checkpoint resnet50_transfer.pt
    uv run python scripts/smoke_predict.py --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import base64
import logging
from collections import Counter
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/raw/kaggle_3m")
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_CHECKPOINT = "resnet50_transfer.pt"
DEFAULT_LIMIT = 20
REQUEST_TIMEOUT_S = 30.0


def find_scan_images(data_dir: Path, limit: int) -> list[Path]:
    """Collect MRI scan images, excluding segmentation masks.

    Args:
        data_dir: Root directory to search recursively for ``.tif`` files.
        limit: Maximum number of images to return.

    Returns:
        Up to ``limit`` image paths whose names do not contain ``_mask``.
    """
    images = [p for p in data_dir.rglob("*.tif") if "_mask" not in p.name]
    return images[:limit]


def predict_image(
    client: httpx.Client,
    image_path: Path,
    *,
    api_url: str = DEFAULT_API_URL,
    checkpoint_name: str = DEFAULT_CHECKPOINT,
) -> dict:
    """Send a single image to the /predict endpoint and return the JSON response.

    Args:
        client: An open httpx client used to issue the request.
        image_path: Path to the image file to classify.
        api_url: Base URL of the running API.
        checkpoint_name: Model checkpoint filename to run inference with.

    Returns:
        The decoded JSON response (see ``PredictionResponse`` schema).

    Raises:
        httpx.HTTPStatusError: If the API returns a non-2xx status.
    """
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    response = client.post(
        f"{api_url}/predict",
        json={"image_base64": img_b64, "checkpoint_name": checkpoint_name},
        timeout=REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


def smoke_predict(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    api_url: str = DEFAULT_API_URL,
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    limit: int = DEFAULT_LIMIT,
) -> list[dict]:
    """Run predictions over a batch of dataset images and print each result.

    Args:
        data_dir: Directory of MRI scans to sample from.
        api_url: Base URL of the running API.
        checkpoint_name: Model checkpoint filename to run inference with.
        limit: Maximum number of images to send.

    Returns:
        A list of prediction JSON responses, one per successfully scored image.
    """
    images = find_scan_images(data_dir, limit)
    if not images:
        logger.warning("No scan images found under %s", data_dir)
        return []

    results: list[dict] = []
    with httpx.Client() as client:
        for image_path in images:
            try:
                result = predict_image(
                    client,
                    image_path,
                    api_url=api_url,
                    checkpoint_name=checkpoint_name,
                )
            except httpx.HTTPError as exc:
                logger.error("%s -> request failed: %s", image_path.name, exc)
                continue
            results.append(result)
            print(f"{image_path.name}: {result['label']} ({result['confidence']:.2f})")

    label_counts = Counter(r["label"] for r in results)
    print(f"\n{len(results)}/{len(images)} scored | distribution: {dict(label_counts)}")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # silence per-request noise
    args = _parse_args()
    smoke_predict(
        data_dir=args.data_dir,
        api_url=args.api_url,
        checkpoint_name=args.checkpoint,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
