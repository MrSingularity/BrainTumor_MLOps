"""Tests for mlops_project.utils.wandb_logging.
 
All tests run in W&B-disabled mode (no real API calls).
Covers wandb_enabled(), wandb_run(), and log_artifact() no-op paths.
"""
 
from __future__ import annotations
 
import os
import pytest
 
from mlops_project.utils.wandb_logging import log_artifact, wandb_enabled, wandb_run
 
 
# ---------------------------------------------------------------------------
# wandb_enabled()
# ---------------------------------------------------------------------------
 
class TestWandbEnabled:
    def test_disabled_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        assert wandb_enabled() is False
 
    def test_disabled_when_mode_is_disabled(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake-key-123")
        monkeypatch.setenv("WANDB_MODE", "disabled")
        assert wandb_enabled() is False
 
    def test_enabled_when_api_key_set(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake-key-123")
        monkeypatch.delenv("WANDB_MODE", raising=False)
        assert wandb_enabled() is True
 
    def test_disabled_when_api_key_empty_string(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "")
        monkeypatch.delenv("WANDB_MODE", raising=False)
        assert wandb_enabled() is False
 
 
# ---------------------------------------------------------------------------
# wandb_run() — disabled path only (no real W&B calls)
# ---------------------------------------------------------------------------
 
class TestWandbRun:
    def test_yields_none_when_disabled(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_MODE", "disabled")
        with wandb_run(job_type="test-job") as run:
            assert run is None
 
    def test_context_manager_completes_without_error(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        ran = False
        with wandb_run(job_type="test-job") as run:
            ran = True
            assert run is None
        assert ran
 
    def test_accepts_all_optional_kwargs(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        with wandb_run(
            project="test-project",
            job_type="test",
            name="test-run",
            config={"lr": 0.001},
        ) as run:
            assert run is None
 
    def test_prints_disabled_message(self, monkeypatch, capsys):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        with wandb_run(job_type="data-prep") as _:
            pass
        captured = capsys.readouterr()
        assert "disabled" in captured.out.lower() or "wandb" in captured.out.lower()
 
 
# ---------------------------------------------------------------------------
# log_artifact() — no-op when run is None
# ---------------------------------------------------------------------------
 
class TestLogArtifact:
    def test_noop_when_run_is_none(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        # Should not raise
        log_artifact("test-artifact", [f], run=None)
 
    def test_noop_with_empty_paths(self):
        log_artifact("test-artifact", [], run=None)
 
    def test_noop_with_multiple_paths(self, tmp_path):
        files = []
        for i in range(3):
            f = tmp_path / f"file_{i}.txt"
            f.write_text(f"content {i}")
            files.append(f)
        log_artifact("multi-artifact", files, run=None)
 
    def test_accepts_custom_artifact_type(self, tmp_path):
        f = tmp_path / "model.pt"
        f.write_bytes(b"fake model")
        log_artifact("my-model", [f], artifact_type="model", run=None)
 
    def test_accepts_description(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.write_bytes(b"fake parquet")
        log_artifact("my-data", [f], description="processed slice index", run=None)
 
    def test_accepts_string_paths(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("a,b\n1,2")
        log_artifact("csv-artifact", [str(f)], run=None)