import os
import json
from pathlib import Path

from mlops_project.api import metrics as metrics_mod
from mlops_project.api import logs_viewer


def test_prometheus_exposition_contains_metrics():
    text = metrics_mod.metrics.prometheus_metrics()
    assert "brain_tumor_api_requests_total" in text
    assert "brain_tumor_api_predictions_total" in text


def test_logs_viewer_drift(tmp_path):
    # Create deterministic small log file
    logs_file = tmp_path / "predictions.jsonl"
    entries = []
    # reference period (first 2): mostly no_tumor
    entries.append({"timestamp": "2026-01-01T00:00:00Z", "label": "no_tumor", "confidence": 0.9, "latency_ms": 50, "model_name": "m1"})
    entries.append({"timestamp": "2026-01-02T00:00:00Z", "label": "no_tumor", "confidence": 0.8, "latency_ms": 60, "model_name": "m1"})
    # current period (next 2): higher tumor rate and higher latency
    entries.append({"timestamp": "2026-02-01T00:00:00Z", "label": "tumor", "confidence": 0.4, "latency_ms": 200, "model_name": "m1"})
    entries.append({"timestamp": "2026-02-02T00:00:00Z", "label": "tumor", "confidence": 0.35, "latency_ms": 210, "model_name": "m1"})

    with open(logs_file, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")

    # Monkeypatch expected location
    orig = logs_viewer.LOGS_FILE
    try:
        logs_viewer.LOGS_FILE = logs_file
        summary = logs_viewer.build_drift_summary()
        assert summary["total_events"] == 4
        assert summary["current_positive_rate"] > summary["reference_positive_rate"]
        assert summary["drift_score"] > 0
    finally:
        logs_viewer.LOGS_FILE = orig
