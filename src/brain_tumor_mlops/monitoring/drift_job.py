"""Scheduled drift job for prediction logs.

The job compares a recent window of real prediction logs against an older
reference window, generates an Evidently report, and writes a compact status
JSON that the API exposes as Prometheus metrics.

Run daily at 02:00 with cron:
    0 2 * * * uv run python -m brain_tumor_mlops.monitoring.drift_job
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from evidently.core.report import Report
from evidently.presets import DataDriftPreset

from brain_tumor_mlops.api.logs_viewer import load_logs

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "data" / "processed" / "drift_reports"
DEFAULT_STATUS_FILE = PROJECT_ROOT / "data" / "processed" / "drift_status.json"

CRON_SCHEDULE = "0 2 * * *"


@dataclass(frozen=True)
class DriftResult:
    """Summary of a single drift run."""

    status: str
    status_code: int
    score: float
    drift_share: float
    number_of_drifted_columns: int
    total_columns: int
    reference_rows: int
    current_rows: int
    failure_rate: float
    generated_at: str
    report_dir: Path
    report_html: Path
    report_json: Path
    reference_csv: Path
    current_csv: Path
    status_file: Path


def _path_from_env(env_name: str, default: Path) -> Path:
    override = os.getenv(env_name)
    return Path(override) if override else default


def get_reports_dir() -> Path:
    """Return the directory where drift reports should be stored."""
    return _path_from_env("MLOPS_DRIFT_REPORTS_DIR", DEFAULT_REPORTS_DIR)


def get_status_file() -> Path:
    """Return the JSON file used to expose the latest drift status."""
    return _path_from_env("MLOPS_DRIFT_STATUS_FILE", DEFAULT_STATUS_FILE)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _select_windows(rows: list[dict[str, Any]], window_size: int = 200) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split logs into reference and current windows using real timestamps."""
    ordered = sorted(
        [row for row in rows if row.get("timestamp")],
        key=lambda row: row.get("timestamp", ""),
    )
    if not ordered:
        return [], []

    if len(ordered) <= window_size:
        midpoint = max(len(ordered) // 2, 1)
        return ordered[:midpoint], ordered[midpoint:]

    current = ordered[-window_size:]
    reference = ordered[-(window_size * 2) : -window_size]
    if not reference:
        midpoint = max(len(ordered) // 2, 1)
        return ordered[:midpoint], ordered[midpoint:]
    return reference, current


def _build_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize raw log rows into a dataframe for Evidently."""
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return frame

    numeric_columns = [
        "confidence",
        "risk_score",
        "latency_ms",
        "threshold",
        "image_mean",
        "image_std",
        "image_height",
        "image_width",
        "image_channels",
    ]
    categorical_columns = ["label", "model_name", "checkpoint_name"]

    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in categorical_columns:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)

    if "timestamp" in frame.columns:
        frame["timestamp"] = frame["timestamp"].astype(str)

    return frame


def _classify_status(
    confidence_delta: float,
    latency_delta_ms: float,
    positive_rate_delta: float,
    failure_rate: float,
) -> tuple[str, int, float]:
    """Map drift indicators to a simple operational status."""
    score = min(
        1.0,
        (min(confidence_delta / 0.25, 1.0) * 0.35)
        + (min(abs(latency_delta_ms) / 250.0, 1.0) * 0.3)
        + (min(abs(positive_rate_delta) / 100.0, 1.0) * 0.2)
        + (min(failure_rate / 0.25, 1.0) * 0.15),
    )
    if score >= 0.6:
        return "ALERT", 2, score
    if score >= 0.3:
        return "WARN", 1, score
    return "OK", 0, score


def run_drift_job(
    *,
    reports_dir: Path | None = None,
    status_file: Path | None = None,
    window_size: int = 200,
) -> DriftResult:
    """Run the drift job end-to-end and write the report plus status file."""
    logs = load_logs(include_test_noise=False)
    successes = [row for row in logs if row.get("label") is not None]
    failures = [row for row in logs if row.get("error")]

    reports_dir = reports_dir or get_reports_dir()
    status_file = status_file or get_status_file()
    reports_dir.mkdir(parents=True, exist_ok=True)
    status_file.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    run_dir = reports_dir / generated_at.replace(":", "-")
    run_dir.mkdir(parents=True, exist_ok=True)

    reference_rows, current_rows = _select_windows(successes, window_size=window_size)
    reference_df = _build_dataframe(reference_rows)
    current_df = _build_dataframe(current_rows)

    reference_csv = run_dir / "reference.csv"
    current_csv = run_dir / "current.csv"
    reference_df.to_csv(reference_csv, index=False)
    current_df.to_csv(current_csv, index=False)

    if reference_df.empty or current_df.empty:
        status = "UNKNOWN"
        status_code = -1
        score = 0.0
        report_html = run_dir / "drift_report.html"
        report_json = run_dir / "drift_report.json"
        report_html.write_text(
            "<html><body><h1>Drift report unavailable</h1>"
            "<p>Insufficient real prediction logs to compute drift.</p></body></html>",
            encoding="utf-8",
        )
    else:
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=reference_df, current_data=current_df)
        report_html = run_dir / "drift_report.html"
        report_json = run_dir / "drift_report.json"
        snapshot.save_html(str(report_html))

        reference_confidences = [float(row.get("confidence", 0.0)) for row in reference_rows]
        current_confidences = [float(row.get("confidence", 0.0)) for row in current_rows]
        reference_latencies = [float(row.get("latency_ms", 0.0)) for row in reference_rows]
        current_latencies = [float(row.get("latency_ms", 0.0)) for row in current_rows]

        def _mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        def _positive_rate(rows: list[dict[str, Any]]) -> float:
            return (
                sum(1 for row in rows if row.get("label") == "tumor") / len(rows) * 100
                if rows
                else 0.0
            )

        confidence_delta = abs(_mean(current_confidences) - _mean(reference_confidences))
        latency_delta_ms = abs(_mean(current_latencies) - _mean(reference_latencies))
        positive_rate_delta = _positive_rate(current_rows) - _positive_rate(reference_rows)
        failure_rate = len(failures) / len(logs) if logs else 0.0
        status, status_code, score = _classify_status(
            confidence_delta,
            latency_delta_ms,
            positive_rate_delta,
            failure_rate,
        )

    summary = {
        "generated_at": generated_at,
        "status": status,
        "status_code": status_code,
        "score": score,
        "drift_share": 0.0,
        "number_of_drifted_columns": 0,
        "total_columns": len(reference_df.columns) if not reference_df.empty else 0,
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
        "failure_rate": len(failures) / len(logs) if logs else 0.0,
        "log_rows": len(logs),
        "successful_rows": len(successes),
        "failed_rows": len(failures),
        "report_dir": str(run_dir),
        "report_html": str(report_html),
        "report_json": str(report_json),
        "reference_csv": str(reference_csv),
        "current_csv": str(current_csv),
        "evidently_report": "DataDriftPreset",
    }

    if reference_df.empty or current_df.empty:
        summary["drift_status"] = "UNKNOWN"
    else:
        summary["confidence_delta"] = confidence_delta
        summary["latency_delta_ms"] = latency_delta_ms
        summary["positive_rate_delta"] = positive_rate_delta
        summary["evidently_snapshot"] = snapshot.dict()

    drift_share = float(summary["drift_share"])
    drifted_columns = int(summary["number_of_drifted_columns"])
    total_columns = int(summary["total_columns"])

    report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    status_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(
        "Drift job finished: status=%s score=%.3f drift_share=%.3f drifted_columns=%s",
        status,
        score,
        drift_share,
        drifted_columns,
    )

    return DriftResult(
        status=status,
        status_code=status_code,
        score=score,
        drift_share=drift_share,
        number_of_drifted_columns=drifted_columns,
        total_columns=total_columns,
        reference_rows=len(reference_df),
        current_rows=len(current_df),
        failure_rate=len(failures) / len(logs) if logs else 0.0,
        generated_at=generated_at,
        report_dir=run_dir,
        report_html=report_html,
        report_json=report_json,
        reference_csv=reference_csv,
        current_csv=current_csv,
        status_file=status_file,
    )


def main() -> None:
    """Command-line entrypoint for cron or manual runs."""
    result = run_drift_job()
    print(
        json.dumps(
            {
                "status": result.status,
                "status_code": result.status_code,
                "score": result.score,
                "drift_share": result.drift_share,
                "report_dir": str(result.report_dir),
                "status_file": str(result.status_file),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
