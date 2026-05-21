"""Metrics and monitoring for API predictions."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGS_DIR = PROJECT_ROOT / "data" / "logs"
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"


def ensure_logs_dir() -> None:
    """Create logs directory if it doesn't exist."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


class PredictionMetrics:
    """Thread-safe metrics collector for API predictions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_state()
        ensure_logs_dir()

    def _reset_state(self) -> None:
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0
        self.total_confidence = 0.0
        self.total_risk_score = 0.0
        self.min_latency_ms = float("inf")
        self.max_latency_ms = 0.0
        self.predictions_by_label = {"tumor": 0, "no_tumor": 0}
        self.models_used: dict[str, int] = {}
        self._registry = CollectorRegistry()
        self._init_prometheus_metrics()

    def _init_prometheus_metrics(self) -> None:
        self.request_counter = Counter(
            "brain_tumor_api_requests_total",
            "Total API requests by status",
            ["status"],
            registry=self._registry,
        )
        self.prediction_counter = Counter(
            "brain_tumor_api_predictions_total",
            "Total successful predictions by class and model",
            ["label", "model_name"],
            registry=self._registry,
        )
        self.latency_histogram = Histogram(
            "brain_tumor_api_inference_latency_seconds",
            "Inference latency in seconds",
            registry=self._registry,
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )
        self.confidence_histogram = Histogram(
            "brain_tumor_api_prediction_confidence",
            "Prediction confidence for successful predictions",
            registry=self._registry,
            buckets=(0.1, 0.25, 0.5, 0.65, 0.8, 0.9, 0.95, 0.99, 1.0),
        )
        self.success_rate_gauge = Gauge(
            "brain_tumor_api_success_rate",
            "Success rate as a percentage",
            registry=self._registry,
        )
        self.positive_rate_gauge = Gauge(
            "brain_tumor_api_positive_prediction_rate",
            "Tumor prediction rate as a percentage",
            registry=self._registry,
        )
        self.last_prediction_timestamp = Gauge(
            "brain_tumor_api_last_prediction_timestamp_seconds",
            "Unix timestamp of the latest prediction event",
            registry=self._registry,
        )
        self.image_mean_histogram = Histogram(
            "brain_tumor_api_image_mean",
            "Mean intensity for incoming prediction images",
            registry=self._registry,
            buckets=(0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
        )
        self.image_std_histogram = Histogram(
            "brain_tumor_api_image_std",
            "Std intensity for incoming prediction images",
            registry=self._registry,
            buckets=(0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3),
        )

    def _refresh_prometheus_snapshot(self) -> None:
        success_rate = (
            self.successful_requests / self.total_requests * 100
            if self.total_requests > 0
            else 0.0
        )
        positive_rate = (
            self.predictions_by_label.get("tumor", 0) / self.successful_requests * 100
            if self.successful_requests > 0
            else 0.0
        )
        self.success_rate_gauge.set(success_rate)
        self.positive_rate_gauge.set(positive_rate)
        self.last_prediction_timestamp.set(datetime.now(timezone.utc).timestamp())

    def log_prediction(
        self,
        label: str,
        confidence: float,
        risk_score: float,
        model_name: str,
        latency_ms: float,
        image_hash: str,
        checkpoint_name: str,
        threshold: float,
        image_stats: dict[str, Any] | None = None,
    ) -> None:
        """Log a prediction to both metrics and JSONL file."""
        payload_stats = dict(image_stats or {})
        safe_label = label if label in {"tumor", "no_tumor"} else "no_tumor"

        with self._lock:
            self.total_requests += 1
            self.successful_requests += 1
            self.total_latency_ms += latency_ms
            self.total_confidence += confidence
            self.total_risk_score += risk_score
            self.min_latency_ms = min(self.min_latency_ms, latency_ms)
            self.max_latency_ms = max(self.max_latency_ms, latency_ms)
            self.predictions_by_label[safe_label] = self.predictions_by_label.get(safe_label, 0) + 1
            self.models_used[model_name] = self.models_used.get(model_name, 0) + 1

            self.request_counter.labels(status="success").inc()
            self.prediction_counter.labels(label=safe_label, model_name=model_name).inc()
            self.latency_histogram.observe(latency_ms / 1000.0)
            self.confidence_histogram.observe(confidence)
            if "image_mean" in payload_stats:
                self.image_mean_histogram.observe(float(payload_stats["image_mean"]))
            if "image_std" in payload_stats:
                self.image_std_histogram.observe(float(payload_stats["image_std"]))
            self._refresh_prometheus_snapshot()

        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": safe_label,
            "confidence": confidence,
            "risk_score": risk_score,
            "model_name": model_name,
            "checkpoint_name": checkpoint_name,
            "latency_ms": latency_ms,
            "image_hash": image_hash,
            "threshold": threshold,
        }
        log_entry.update(payload_stats)

        try:
            with open(PREDICTIONS_LOG, "a", encoding="utf-8") as file_handle:
                file_handle.write(json.dumps(log_entry) + "\n")
        except Exception as e:  # pragma: no cover - logging must not break inference
            logger.error(f"Failed to write prediction log: {e}")

    def log_error(self, checkpoint_name: str, error_message: str) -> None:
        """Log a failed prediction."""
        with self._lock:
            self.total_requests += 1
            self.failed_requests += 1
            self.request_counter.labels(status="failed").inc()
            self._refresh_prometheus_snapshot()

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint_name": checkpoint_name,
            "error": error_message,
        }

        try:
            with open(PREDICTIONS_LOG, "a", encoding="utf-8") as file_handle:
                file_handle.write(json.dumps(log_entry) + "\n")
        except Exception as e:  # pragma: no cover - logging must not break inference
            logger.error(f"Failed to write error log: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Return current metrics snapshot."""
        with self._lock:
            avg_latency = (
                self.total_latency_ms / self.successful_requests
                if self.successful_requests > 0
                else 0.0
            )
            avg_confidence = (
                self.total_confidence / self.successful_requests
                if self.successful_requests > 0
                else 0.0
            )
            avg_risk_score = (
                self.total_risk_score / self.successful_requests
                if self.successful_requests > 0
                else 0.0
            )
            positive_rate = (
                self.predictions_by_label.get("tumor", 0) / self.successful_requests * 100
                if self.successful_requests > 0
                else 0.0
            )
            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": (
                    self.successful_requests / self.total_requests * 100
                    if self.total_requests > 0
                    else 0.0
                ),
                "positive_prediction_rate": positive_rate,
                "latency_ms": {
                    "min": self.min_latency_ms if self.min_latency_ms != float("inf") else 0.0,
                    "max": self.max_latency_ms,
                    "avg": avg_latency,
                },
                "average_confidence": avg_confidence,
                "average_risk_score": avg_risk_score,
                "predictions_by_label": dict(self.predictions_by_label),
                "models_used": dict(self.models_used),
            }

    def prometheus_metrics(self) -> str:
        """Return the Prometheus exposition format for the current registry."""
        return generate_latest(self._registry).decode("utf-8")

    @property
    def prometheus_content_type(self) -> str:
        """Return the Prometheus content type header."""
        return CONTENT_TYPE_LATEST

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._reset_state()


# Global metrics instance
metrics = PredictionMetrics()
