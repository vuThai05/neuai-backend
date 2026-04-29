"""Run retrieval over a list of queries and print score histograms.

Useful for sanity-checking retrieval quality after re-indexing or
threshold tuning. Requires the full backend stack (Mongo + Qdrant +
BGE-M3) to be reachable, so it should be run as a manual diagnostic.

Usage:
    python scripts/evaluate_retrieval.py queries.txt [--top-k 5]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()


def histogram(scores: List[float], buckets: int = 10) -> Counter:
    counts: Counter = Counter()
    for s in scores:
        bucket = min(int(s * buckets), buckets - 1)
        counts[f"{bucket / buckets:.2f}-{(bucket + 1) / buckets:.2f}"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries_file", help="Plain text file: one query per line")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    from src.rag.retriever import RAGRetriever
    from src.rag.scoring import compute_signals

    retriever = RAGRetriever()

    queries = [line.strip() for line in Path(args.queries_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not queries:
        print("No queries to evaluate.", file=sys.stderr)
        return 1

    top_scores: List[float] = []
    avg_top3_list: List[float] = []
    diversities: List[int] = []

    for q in queries:
        docs = retriever.retrieve(q, top_k=args.top_k) or []
        signals = compute_signals(docs, q)
        top_scores.append(signals.top_score)
        avg_top3_list.append(signals.avg_top3)
        diversities.append(signals.source_diversity)
        print(
            f"q={q[:50]!r}  top={signals.top_score:.3f}  "
            f"avg3={signals.avg_top3:.3f}  diversity={signals.source_diversity}  "
            f"freshness={signals.needs_freshness}"
        )

    print("\nTop score histogram:")
    for bucket, n in sorted(histogram(top_scores).items()):
        print(f"  {bucket}: {'#' * n}  ({n})")

    print("\nSummary:")
    print(f"  queries={len(queries)}")
    print(f"  top_score mean={statistics.mean(top_scores):.3f} median={statistics.median(top_scores):.3f}")
    print(f"  avg_top3 mean={statistics.mean(avg_top3_list):.3f}")
    print(f"  diversity mean={statistics.mean(diversities):.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
