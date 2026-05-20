"""Utility to view and analyze prediction logs."""

import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGS_FILE = PROJECT_ROOT / "data" / "logs" / "predictions.jsonl"


def load_logs() -> list[dict]:
    """Load all prediction logs from JSONL file."""
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
    return logs


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
