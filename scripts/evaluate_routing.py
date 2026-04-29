"""Run the decision engine over a list of (query, expected_route) pairs.

Designed for offline tuning of the policy thresholds: you can tweak
`HIGH_TOP_SCORE` etc. and rerun this script to see the confusion matrix.

The dataset format is JSON:
    [
      {"question": "...", "expected_route": "RAG_ONLY"|"RAG_PLUS_WEB",
       "docs": [{"score": 0.9, "text": "...", "source": {...}}, ...]},
      ...
    ]

Usage:
    python scripts/evaluate_routing.py path/to/dataset.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any, Dict, List

# Allow running as a module from the repo root or as a script from anywhere.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src.orchestration.decision_engine import decide
from src.rag.scoring import compute_signals


def evaluate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    confusion: Counter = Counter()
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        signals = compute_signals(sample.get("docs", []), sample.get("question", ""))
        decision = decide(signals)
        actual = decision.route
        expected = sample.get("expected_route")
        confusion[(expected, actual)] += 1
        rows.append({
            "question": sample.get("question", "")[:60],
            "expected": expected,
            "actual": actual,
            "reason_code": decision.reason_code,
            "top_score": signals.top_score,
            "avg_top3": signals.avg_top3,
            "diversity": signals.source_diversity,
            "freshness": signals.needs_freshness,
        })

    correct = sum(c for (exp, act), c in confusion.items() if exp == act)
    total = sum(confusion.values())
    return {
        "accuracy": (correct / total) if total else None,
        "confusion": {f"{exp}->{act}": c for (exp, act), c in confusion.items()},
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to JSON dataset")
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as fh:
        samples = json.load(fh)

    result = evaluate(samples)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
