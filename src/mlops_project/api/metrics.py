"""Metrics and monitoring for API predictions."""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGS_DIR = PROJECT_ROOT / "data" / "logs"
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"


def ensure_logs_dir():
    """Create logs directory if it doesn't exist."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


class PredictionMetrics:
    """Thread-safe metrics collector for API predictions."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0
        self.min_latency_ms = float('inf')
        self.max_latency_ms = 0.0
        self.predictions_by_label = {"tumor": 0, "no_tumor": 0}
        self.models_used = {}
        ensure_logs_dir()

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
    ) -> None:
        """Log a prediction to both metrics and JSONL file."""
        with self._lock:
            self.total_requests += 1
            self.successful_requests += 1
            self.total_latency_ms += latency_ms
            self.min_latency_ms = min(self.min_latency_ms, latency_ms)
            self.max_latency_ms = max(self.max_latency_ms, latency_ms)
            
            # Update label counts
            self.predictions_by_label[label] = self.predictions_by_label.get(label, 0) + 1
            
            # Update model usage
            self.models_used[model_name] = self.models_used.get(model_name, 0) + 1

        # Write to JSONL log file
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "label": label,
            "confidence": confidence,
            "risk_score": risk_score,
            "model_name": model_name,
            "checkpoint_name": checkpoint_name,
            "latency_ms": latency_ms,
            "image_hash": image_hash,
            "threshold": threshold,
        }
        
        try:
            with open(PREDICTIONS_LOG, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write prediction log: {e}")

    def log_error(self, checkpoint_name: str, error_message: str) -> None:
        """Log a failed prediction."""
        with self._lock:
            self.total_requests += 1
            self.failed_requests += 1

        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checkpoint_name": checkpoint_name,
            "error": error_message,
        }
        
        try:
            with open(PREDICTIONS_LOG, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write error log: {e}")

    def get_metrics(self) -> dict:
        """Return current metrics snapshot."""
        with self._lock:
            avg_latency = (
                self.total_latency_ms / self.successful_requests
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
                "latency_ms": {
                    "min": self.min_latency_ms if self.min_latency_ms != float('inf') else 0.0,
                    "max": self.max_latency_ms,
                    "avg": avg_latency,
                },
                "predictions_by_label": dict(self.predictions_by_label),
                "models_used": dict(self.models_used),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.total_latency_ms = 0.0
            self.min_latency_ms = float('inf')
            self.max_latency_ms = 0.0
            self.predictions_by_label = {"tumor": 0, "no_tumor": 0}
            self.models_used = {}


# Global metrics instance
metrics = PredictionMetrics()
