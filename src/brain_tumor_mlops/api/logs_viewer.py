"""Utility to view and analyze prediction logs."""

import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGS_FILE = PROJECT_ROOT / "data" / "logs" / "predictions.jsonl"
TEST_CHECKPOINT_NAMES = {"missing.pt", "nonexistent.pt"}
TEST_ERROR_PREFIXES = (
    "Invalid base64",
    "Empty image data",
    "Image too small",
)


def _is_test_noise(log: dict) -> bool:
    """Return True for synthetic test failures that should not dominate ops views."""
    checkpoint_name = str(log.get("checkpoint_name", ""))
    error_message = str(log.get("error", ""))
    if checkpoint_name in TEST_CHECKPOINT_NAMES:
        return True
    return any(error_message.startswith(prefix) for prefix in TEST_ERROR_PREFIXES)


def load_logs(include_test_noise: bool = False) -> list[dict]:
    """Load prediction logs from JSONL file.

    By default, synthetic failures from the test suite are excluded so the ops
    dashboard stays focused on real application activity.
    """
    if not LOGS_FILE.exists():
        return []
    
    logs = []
    with open(LOGS_FILE, "r") as f:
        for line in f:
            if line.strip():
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if include_test_noise:
        return logs
    return [log for log in logs if not _is_test_noise(log)]


def summarize_logs(logs: list[dict] | None = None) -> dict:
    """Summarize prediction logs for operational dashboards."""
    entries = logs if logs is not None else load_logs()
    successes = [log for log in entries if "label" in log]
    failures = [log for log in entries if "error" in log]
    confidences = [float(log.get("confidence", 0.0)) for log in successes]
    latencies = [float(log.get("latency_ms", 0.0)) for log in successes]
    positive_rate = (
        sum(1 for log in successes if log.get("label") == "tumor") / len(successes) * 100
        if successes
        else 0.0
    )
    by_model = defaultdict(int)
    by_label = defaultdict(int)
    for log in successes:
        by_model[log.get("model_name", "unknown")] += 1
        by_label[log.get("label", "unknown")] += 1
    return {
        "total_events": len(entries),
        "successful_events": len(successes),
        "failed_events": len(failures),
        "success_rate": len(successes) / len(entries) * 100 if entries else 0.0,
        "positive_prediction_rate": positive_rate,
        "average_confidence": mean(confidences) if confidences else 0.0,
        "average_latency_ms": mean(latencies) if latencies else 0.0,
        "predictions_by_model": dict(by_model),
        "predictions_by_label": dict(by_label),
        "recent_successes": successes[-5:],
        "recent_failures": failures[-5:],
    }


def build_drift_summary(logs: list[dict] | None = None) -> dict:
    """Build a drift summary by comparing recent and previous log windows."""
    entries = logs if logs is not None else load_logs()
    if not entries:
        return {
            "total_events": 0,
            "window_size": 0,
            "reference_positive_rate": 0.0,
            "current_positive_rate": 0.0,
            "reference_failure_rate": 0.0,
            "current_failure_rate": 0.0,
            "positive_rate_delta": 0.0,
            "failure_rate_delta": 0.0,
            "confidence_delta": 0.0,
            "latency_delta_ms": 0.0,
            "drift_score": 0.0,
        }

    window_size = max(2, len(entries) // 4)
    window_size = min(window_size, max(len(entries) // 2, 1))
    reference_window = entries[:-window_size] or entries[:window_size]
    current_window = entries[-window_size:]

    reference = [log for log in reference_window if "label" in log]
    current = [log for log in current_window if "label" in log] or reference
    reference_failures = [log for log in reference_window if "error" in log]
    current_failures = [log for log in current_window if "error" in log]

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _positive_rate(rows: list[dict]) -> float:
        return (
            sum(1 for row in rows if row.get("label") == "tumor") / len(rows) * 100
            if rows
            else 0.0
        )

    def _failure_rate(rows: list[dict]) -> float:
        return sum(1 for row in rows if "error" in row) / len(rows) * 100 if rows else 0.0

    reference_confidences = [float(row.get("confidence", 0.0)) for row in reference]
    current_confidences = [float(row.get("confidence", 0.0)) for row in current]
    reference_latencies = [float(row.get("latency_ms", 0.0)) for row in reference]
    current_latencies = [float(row.get("latency_ms", 0.0)) for row in current]
    reference_positive_rate = _positive_rate(reference)
    current_positive_rate = _positive_rate(current)
    reference_failure_rate = _failure_rate(reference_window)
    current_failure_rate = _failure_rate(current_window)
    confidence_delta = abs(_mean(current_confidences) - _mean(reference_confidences))
    latency_delta_ms = abs(_mean(current_latencies) - _mean(reference_latencies))
    positive_rate_delta = current_positive_rate - reference_positive_rate
    failure_rate_delta = current_failure_rate - reference_failure_rate

    confidence_component = min(confidence_delta / 0.25, 1.0)
    latency_component = min(latency_delta_ms / 250.0, 1.0)
    positive_component = min(abs(positive_rate_delta) / 100.0, 1.0)
    failure_component = min(abs(failure_rate_delta) / 100.0, 1.0)
    drift_score = min(
        1.0,
        0.35 * confidence_component
        + 0.30 * latency_component
        + 0.20 * positive_component
        + 0.15 * failure_component,
    )

    return {
        "total_events": len(entries),
        "window_size": window_size,
        "reference_events": len(reference_window),
        "current_events": len(current_window),
        "reference_positive_rate": reference_positive_rate,
        "current_positive_rate": current_positive_rate,
        "positive_rate_delta": positive_rate_delta,
        "reference_failure_rate": reference_failure_rate,
        "current_failure_rate": current_failure_rate,
        "failure_rate_delta": failure_rate_delta,
        "confidence_delta": confidence_delta,
        "latency_delta_ms": latency_delta_ms,
        "drift_score": drift_score,
    }


def analyze_logs():
    """Print analysis of prediction logs."""
    logs = load_logs()
    
    if not logs:
        print("❌ No logs found in data/logs/predictions.jsonl")
        return
    
    print(f"\n{'='*80}")
    print(f"PREDICTION LOG ANALYSIS — {len(logs)} entries")
    print(f"{'='*80}\n")
    
    # Separate successes and failures
    successes = [log for log in logs if "label" in log]
    failures = [log for log in logs if "error" in log]
    
    print(f"✓ Successful predictions: {len(successes)}")
    print(f"✗ Failed predictions:     {len(failures)}")
    print(f"Success rate: {len(successes) / len(logs) * 100:.1f}%\n")
    
    # Analyze successful predictions
    if successes:
        print(f"{'─'*80}")
        print("SUCCESSFUL PREDICTIONS")
        print(f"{'─'*80}")
        
        by_label = defaultdict(int)
        by_model = defaultdict(int)
        latencies = []
        confidences = []
        
        for log in successes:
            by_label[log.get("label", "unknown")] += 1
            by_model[log.get("model_name", "unknown")] += 1
            latencies.append(log.get("latency_ms", 0))
            confidences.append(log.get("confidence", 0))
        
        print(f"\nPredictions by class:")
        for label, count in sorted(by_label.items()):
            print(f"  • {label.upper()}: {count}")
        
        print(f"\nModels used:")
        for model, count in sorted(by_model.items(), key=lambda x: -x[1]):
            print(f"  • {model}: {count} predictions")
        
        if latencies:
            print(f"\nLatency stats (ms):")
            print(f"  • Min:  {min(latencies):.1f}ms")
            print(f"  • Max:  {max(latencies):.1f}ms")
            print(f"  • Avg:  {sum(latencies) / len(latencies):.1f}ms")
        
        if confidences:
            print(f"\nConfidence stats:")
            print(f"  • Min:  {min(confidences):.3f}")
            print(f"  • Max:  {max(confidences):.3f}")
            print(f"  • Avg:  {sum(confidences) / len(confidences):.3f}")
        
        print(f"\nRecent successes (last 5):")
        for log in successes[-5:]:
            ts = log.get("timestamp", "?")
            label = log.get("label", "?").upper()
            conf = log.get("confidence", 0)
            latency = log.get("latency_ms", 0)
            model = log.get("model_name", "?")
            print(f"  [{ts}] {label} @ {conf:.1%} confidence ({latency:.0f}ms, {model})")
    
    # Analyze failures
    if failures:
        print(f"\n{'─'*80}")
        print("FAILED PREDICTIONS")
        print(f"{'─'*80}")
        
        error_types = defaultdict(int)
        for log in failures:
            error = log.get("error", "unknown error")
            # Extract error type (first 50 chars)
            error_type = error[:50] + "..." if len(error) > 50 else error
            error_types[error_type] += 1
        
        print(f"\nError types:")
        for error, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  • {error}: {count}x")
        
        print(f"\nRecent failures (last 5):")
        for log in failures[-5:]:
            ts = log.get("timestamp", "?")
            error = log.get("error", "?")
            error_short = error[:60] + "..." if len(error) > 60 else error
            print(f"  [{ts}] ✗ {error_short}")
    
    print(f"\n{'='*80}\n")


def show_recent(n: int = 10):
    """Show recent N log entries."""
    logs = load_logs()
    
    if not logs:
        print("❌ No logs found")
        return
    
    print(f"\n{'='*80}")
    print(f"RECENT LOGS (last {n} entries)")
    print(f"{'='*80}\n")
    
    for log in logs[-n:]:
        if "label" in log:
            # Success
            print(f"✓ [{log['timestamp']}]")
            print(f"  Result: {log['label'].upper()} @ {log['confidence']:.1%}")
            print(f"  Model: {log['model_name']} | Latency: {log['latency_ms']:.0f}ms")
            print(f"  Image: {log['image_hash']} | Threshold: {log['threshold']}")
        else:
            # Failure
            print(f"✗ [{log['timestamp']}]")
            print(f"  Error: {log['error']}")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--recent":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        show_recent(n)
    else:
        analyze_logs()
