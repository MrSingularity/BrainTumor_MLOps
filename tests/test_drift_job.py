from __future__ import annotations

import json
from pathlib import Path

import brain_tumor_mlops.api.logs_viewer as logs_viewer
import brain_tumor_mlops.api.metrics as metrics_mod
from brain_tumor_mlops.monitoring.drift_job import run_drift_job


def _write_logs(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _make_row(timestamp: str, label: str, confidence: float, latency_ms: float, model_name: str) -> dict:
    return {
        "timestamp": timestamp,
        "label": label,
        "confidence": confidence,
        "risk_score": confidence,
        "model_name": model_name,
        "checkpoint_name": f"{model_name}.pt",
        "latency_ms": latency_ms,
        "image_hash": "abc123",
        "threshold": 0.5,
        "image_mean": 0.4,
        "image_std": 0.1,
        "image_height": 256,
        "image_width": 256,
        "image_channels": 3,
    }


def test_run_drift_job_writes_report_and_status(tmp_path, monkeypatch):
    log_file = tmp_path / "predictions.jsonl"
    rows = []
    for idx in range(20):
        rows.append(_make_row(f"2026-05-01T00:{idx:02d}:00Z", "no_tumor", 0.92, 40 + idx, "baseline"))
    for idx in range(20):
        rows.append(_make_row(f"2026-05-20T00:{idx:02d}:00Z", "tumor", 0.35, 180 + idx, "resnet50_transfer"))
    _write_logs(log_file, rows)

    monkeypatch.setattr(logs_viewer, "LOGS_FILE", log_file)
    monkeypatch.setenv("MLOPS_DRIFT_STATUS_FILE", str(tmp_path / "drift_status.json"))

    result = run_drift_job(reports_dir=tmp_path / "reports")

    assert result.report_html.exists()
    assert result.report_json.exists()
    assert result.reference_csv.exists()
    assert result.current_csv.exists()
    assert result.status in {"OK", "WARN", "ALERT", "UNKNOWN"}

    status_payload = json.loads(result.status_file.read_text(encoding="utf-8"))
    assert status_payload["log_rows"] == 40
    assert status_payload["generated_at"]
    assert "report_html" in status_payload

    prometheus_text = metrics_mod.metrics.prometheus_metrics()
    assert "brain_tumor_api_drift_status" in prometheus_text
    assert "brain_tumor_api_drift_score" in prometheus_text
    assert "brain_tumor_api_drift_status_code" in prometheus_text
