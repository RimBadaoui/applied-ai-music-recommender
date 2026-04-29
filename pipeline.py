"""
pipeline.py — Orchestrates the full RAG recommendation pipeline.

Flow:
  User query (str)
      │
      ▼
  [Guardrail] validate_query
      │
      ▼
  [query_parser] parse_query  ──► structured prefs dict
      │
      ▼
  [Guardrail] validate_parsed_prefs
      │
      ▼
  [catalog] retrieve_candidates  ──► top-K songs (cosine sim)
      │
      ▼
  [ranker] rank_and_explain  ──► re-ranked + explained results (Claude)
      │
      ▼
  [Guardrail] validate_results
      │
      ▼
  PipelineResult  (printed / returned to caller)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from catalog import load_songs, retrieve_candidates
from guardrails import validate_parsed_prefs, validate_query, validate_results
from query_parser import parse_query
from ranker import rank_and_explain

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    query: str
    parsed_prefs: Dict
    recommendations: List[Dict]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and len(self.recommendations) > 0

    @property
    def overall_confidence(self) -> float:
        """Mean confidence across all returned recommendations."""
        confs = [r.get("confidence", 0.0) for r in self.recommendations]
        return round(sum(confs) / len(confs), 3) if confs else 0.0


def run(
    user_query: str,
    songs: List[Dict],
    retrieval_k: int = 8,
    final_k: int = 5,
) -> PipelineResult:
    """
    Run the full RAG pipeline for a single user query.

    Args:
        user_query:  Free-text natural-language music request.
        songs:       Full song catalog (list of dicts from load_songs()).
        retrieval_k: How many candidates to pull in the retrieval stage.
        final_k:     Max number of final recommendations to return.

    Returns:
        PipelineResult with recommendations, warnings, errors, and timing.
    """
    start = time.perf_counter()
    all_warnings: List[str] = []
    all_errors: List[str] = []

    # ── Stage 0: validate raw query ─────────────────────────────────────────
    qv = validate_query(user_query)
    if not qv:
        return PipelineResult(
            query=user_query,
            parsed_prefs={},
            recommendations=[],
            errors=qv.errors,
            elapsed_seconds=time.perf_counter() - start,
        )
    all_warnings.extend(qv.warnings)

    # ── Stage 1: parse query → structured prefs ──────────────────────────────
    logger.info("=== Stage 1: Query Parsing ===")
    prefs = parse_query(user_query)

    pv = validate_parsed_prefs(prefs)
    all_warnings.extend(pv.warnings)
    if not pv:
        return PipelineResult(
            query=user_query,
            parsed_prefs=prefs,
            recommendations=[],
            errors=pv.errors,
            warnings=all_warnings,
            elapsed_seconds=time.perf_counter() - start,
        )

    # ── Stage 2: retrieve candidates (cosine similarity RAG) ─────────────────
    logger.info("=== Stage 2: Retrieval ===")
    candidates = retrieve_candidates(prefs, songs, top_k=retrieval_k)
    if not candidates:
        all_errors.append("Retrieval stage returned no candidates.")
        return PipelineResult(
            query=user_query,
            parsed_prefs=prefs,
            recommendations=[],
            errors=all_errors,
            warnings=all_warnings,
            elapsed_seconds=time.perf_counter() - start,
        )

    # ── Stage 3: re-rank + explain with Claude ────────────────────────────────
    logger.info("=== Stage 3: Ranking & Explanation ===")
    results = rank_and_explain(user_query, candidates, k=final_k)

    rv = validate_results(results)
    all_warnings.extend(rv.warnings)
    if not rv:
        all_errors.extend(rv.errors)

    elapsed = round(time.perf_counter() - start, 2)
    logger.info(
        "Pipeline complete. success=%s results=%d elapsed=%.2fs",
        rv.valid,
        len(results),
        elapsed,
    )

    return PipelineResult(
        query=user_query,
        parsed_prefs=prefs,
        recommendations=results,
        warnings=all_warnings,
        errors=all_errors,
        elapsed_seconds=elapsed,
    )


def print_result(result: PipelineResult) -> None:
    """Pretty-print a PipelineResult to stdout."""
    print(f"\n{'━'*60}")
    print(f"  Query   : {result.query}")
    print(f"  Parsed  : genre={result.parsed_prefs.get('genre')}, "
          f"mood={result.parsed_prefs.get('mood')}, "
          f"energy={result.parsed_prefs.get('energy')}")
    print(f"  Parse confidence : {result.parsed_prefs.get('confidence', '—')}")
    print(f"  Overall confidence: {result.overall_confidence:.2f}")
    print(f"  Time    : {result.elapsed_seconds}s")

    if result.warnings:
        for w in result.warnings:
            print(f"  ⚠  {w}")

    if result.errors:
        print(f"\n  ✗ Errors:")
        for e in result.errors:
            print(f"    • {e}")
        print()
        return

    if not result.recommendations:
        print("\n  No recommendations returned.\n")
        return

    print(f"\n  Top {len(result.recommendations)} recommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"\n  {i}. {rec['title']} — {rec['artist']}")
        print(f"     Match score : {rec.get('match_score', '—'):.2f}")
        print(f"     Confidence  : {rec.get('confidence', '—'):.2f}")
        print(f"     Why         : {rec.get('explanation', '')}")
    print()
