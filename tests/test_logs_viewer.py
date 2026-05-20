"""Tests for mlops_project.api.logs_viewer.

Uses tmp_path to create synthetic JSONL log files.
Covers load_logs() and analyze_logs().
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
import unittest.mock as mock

import mlops_project.api.logs_viewer as logs_viewer
from mlops_project.api.logs_viewer import load_logs, analyze_logs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_logs(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


SAMPLE_SUCCESS = {
    "timestamp": "2024-01-01T00:00:00",
    "label": "tumor",
    "confidence": 0.92,
    "model_version": "v1.0",
    "latency_ms": 120.5,
}

SAMPLE_FAILURE = {
    "timestamp": "2024-01-01T00:01:00",
    "error": "invalid image format",
}


# ---------------------------------------------------------------------------
# load_logs()
# ---------------------------------------------------------------------------

class TestLoadLogs:
    def test_returns_empty_list_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(logs_viewer, "LOGS_FILE", tmp_path / "nonexistent.jsonl")
        result = load_logs()
        assert result == []

    def test_returns_list_of_dicts(self, tmp_path, monkeypatch):
        log_file = tmp_path / "predictions.jsonl"
        _write_logs(log_file, [SAMPLE_SUCCESS])
        monkeypatch.setattr(logs_viewer, "LOGS_FILE", log_file)
        result = load_logs()
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_correct_count(self, tmp_path, monkeypatch):
        log_file = tmp_path / "predictions.jsonl"
        _write_logs(log_file, [SAMPLE_SUCCESS, SAMPLE_FAILURE, SAMPLE_SUCCESS])
        monkeypatch.setattr(logs_viewer, "LOGS_FILE", log_file)
        result = load_logs()
        assert len(result) == 3

    def test_skips_invalid_json_lines(self, tmp_path, monkeypatch):
        log_file = tmp_path / "predictions.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            f.write(json.dumps(SAMPLE_SUCCESS) + "\n")
            f.write("NOT VALID JSON\n")
            f.write(json.dumps(SAMPLE_FAILURE) + "\n")
        monkeypatch.setattr(logs_viewer, "LOGS_FILE", log_file)
        result = load_logs()
        assert len(result) == 2

    def test_skips_empty_lines(self, tmp_path, monkeypatch):
        log_file = tmp_path / "predictions.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            f.write(json.dumps(SAMPLE_SUCCESS) + "\n")
            f.write("\n")
            f.write("   \n")
            f.write(json.dumps(SAMPLE_FAILURE) + "\n")
        monkeypatch.setattr(logs_viewer, "LOGS_FILE", log_file)
        result = load_logs()
        assert len(result) == 2

    def test_preserves_log_content(self, tmp_path, monkeypatch):
        log_file = tmp_path / "predictions.jsonl"
        _write_logs(log_file, [SAMPLE_SUCCESS])
        monkeypatch.setattr(logs_viewer, "LOGS_FILE", log_file)
        result = load_logs()
        assert result[0]["label"] == "tumor"
        assert result[0]["confidence"] == pytest.approx(0.92)

    def test_empty_file_returns_empty_list(self, tmp_path, monkeypatch):
        log_file = tmp_path / "predictions.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("")
        monkeypatch.setattr(logs_viewer, "LOGS_FILE", log_file)
        result = load_logs()
        assert result == []


# ---------------------------------------------------------------------------
# analyze_logs()
# ---------------------------------------------------------------------------

class TestAnalyzeLogs:
    def test_prints_no_logs_message_when_empty(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(logs_viewer, "LOGS_FILE", tmp_path / "nonexistent.jsonl")
        analyze_logs()
        captured = capsys.readouterr()
        assert "No logs" in captured.out or "no logs" in captured.out.lower()

    def test_runs_without_error_with_successes(self, tmp_path, monkeypatch):
        log_file = tmp_path / "predictions.jsonl"
        _write_logs(log_file, [SAMPLE_SUCCESS, SAMPLE_SUCCESS])
        monkeypatch.setattr(logs_viewer, "LOGS_FILE", log_file)
        analyze_logs()  # should not raise

    def test_runs_without_error_with_failures(self, tmp_path, monkeypatch):
        log_file = tmp_path / "predictions.jsonl"
        _write_logs(log_file, [SAMPLE_FAILURE])
        monkeypatch.setattr(logs_viewer, "LOGS_FILE", log_file)
        analyze_logs()  # should not raise

    def test_runs_without_error_with_mixed_logs(self, tmp_path, monkeypatch):
        log_file = tmp_path / "predictions.jsonl"
        _write_logs(log_file, [SAMPLE_SUCCESS, SAMPLE_FAILURE, SAMPLE_SUCCESS])
        monkeypatch.setattr(logs_viewer, "LOGS_FILE", log_file)
        analyze_logs()  # should not raise

    def test_output_contains_count(self, tmp_path, monkeypatch, capsys):
        log_file = tmp_path / "predictions.jsonl"
        _write_logs(log_file, [SAMPLE_SUCCESS, SAMPLE_SUCCESS, SAMPLE_FAILURE])
        monkeypatch.setattr(logs_viewer, "LOGS_FILE", log_file)
        analyze_logs()
        captured = capsys.readouterr()
        assert "3" in captured.out or "2" in captured.out