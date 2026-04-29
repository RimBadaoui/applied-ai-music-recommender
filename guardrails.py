"""
guardrails.py — Input validation and output safety checks.

Validates:
  - User queries (not empty, not gibberish, reasonable length)
  - Parsed preference dicts (numeric fields in range, known enum values)
  - Ranked outputs (scores in range, explanations non-empty)

Returns structured ValidationResult objects so callers can decide
whether to proceed, warn, or abort.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

KNOWN_GENRES = {"lofi", "pop", "rock", "jazz", "ambient", "synthwave", "indie pop"}
KNOWN_MOODS = {"happy", "chill", "intense", "moody", "relaxed", "focused"}

MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 500


@dataclass
class ValidationResult:
    valid: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


def validate_query(query: str) -> ValidationResult:
    """Check that the raw user query is usable."""
    warnings: List[str] = []
    errors: List[str] = []

    if not query or not query.strip():
        errors.append("Query is empty.")
        return ValidationResult(valid=False, errors=errors)

    q = query.strip()

    if len(q) < MIN_QUERY_LENGTH:
        errors.append(f"Query too short (min {MIN_QUERY_LENGTH} chars).")

    if len(q) > MAX_QUERY_LENGTH:
        errors.append(f"Query too long (max {MAX_QUERY_LENGTH} chars). Got {len(q)}.")

    # Detect likely gibberish: no vowels, all symbols, etc.
    letters = re.sub(r"[^a-zA-Z]", "", q)
    if len(letters) < 2:
        errors.append("Query appears to contain no readable text.")

    if errors:
        logger.warning("Query validation failed: %s | query=%r", errors, query)
        return ValidationResult(valid=False, warnings=warnings, errors=errors)

    logger.debug("Query validated OK: %r", query)
    return ValidationResult(valid=True, warnings=warnings)


def validate_parsed_prefs(prefs: Dict) -> ValidationResult:
    """Check that the parsed preference dict from query_parser is sane."""
    warnings: List[str] = []
    errors: List[str] = []

    if "_error" in prefs:
        errors.append(f"Parser error: {prefs['_error']}")
        return ValidationResult(valid=False, errors=errors)

    confidence = prefs.get("confidence", 0.0)
    if confidence < 0.25:
        warnings.append(
            f"Low parse confidence ({confidence:.2f}). "
            "Results may not match your request well."
        )

    # Validate numeric ranges
    float_fields = {
        "energy": (0.0, 1.0),
        "valence": (0.0, 1.0),
        "danceability": (0.0, 1.0),
        "acousticness": (0.0, 1.0),
        "tempo_bpm": (40.0, 220.0),
    }
    for key, (lo, hi) in float_fields.items():
        if key in prefs:
            val = prefs[key]
            if not isinstance(val, (int, float)):
                errors.append(f"Field '{key}' should be a number, got {type(val).__name__}.")
            elif not (lo <= float(val) <= hi):
                warnings.append(f"Field '{key}'={val} is outside expected range [{lo}, {hi}].")

    # Validate enum values (warn, don't error — novel genres are okay)
    if "genre" in prefs and prefs["genre"] not in KNOWN_GENRES:
        warnings.append(
            f"Genre '{prefs['genre']}' is not in the catalog. "
            f"Known genres: {', '.join(sorted(KNOWN_GENRES))}."
        )

    if "mood" in prefs and prefs["mood"] not in KNOWN_MOODS:
        warnings.append(
            f"Mood '{prefs['mood']}' is not in the catalog. "
            f"Known moods: {', '.join(sorted(KNOWN_MOODS))}."
        )

    valid = len(errors) == 0
    if warnings:
        logger.warning("Pref validation warnings: %s", warnings)
    if errors:
        logger.error("Pref validation errors: %s", errors)

    return ValidationResult(valid=valid, warnings=warnings, errors=errors)


def validate_results(results: List[Dict]) -> ValidationResult:
    """Sanity-check the ranked output from the ranker."""
    warnings: List[str] = []
    errors: List[str] = []

    if not results:
        errors.append("Ranker returned no results.")
        return ValidationResult(valid=False, errors=errors)

    for i, r in enumerate(results):
        if not r.get("explanation", "").strip():
            warnings.append(f"Result #{i+1} '{r.get('title')}' has an empty explanation.")

        score = r.get("match_score")
        if score is not None and not (0.0 <= float(score) <= 1.0):
            warnings.append(f"Result #{i+1} has out-of-range match_score={score}.")

        conf = r.get("confidence")
        if conf is not None and float(conf) < 0.3:
            warnings.append(
                f"Result #{i+1} '{r.get('title')}' has low confidence ({conf:.2f})."
            )

    valid = len(errors) == 0
    if warnings:
        logger.warning("Result validation warnings: %s", warnings)
    if errors:
        logger.error("Result validation errors: %s", errors)

    return ValidationResult(valid=valid, warnings=warnings, errors=errors)
