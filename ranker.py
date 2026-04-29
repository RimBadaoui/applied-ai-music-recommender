"""
ranker.py — Stage 2 of the RAG pipeline.

Takes the top-K candidate songs retrieved by catalog.retrieve_candidates()
and the original user query, then asks Claude to:
  1. Re-rank the candidates by relevance to the natural-language query.
  2. Write a plain-English explanation for each recommendation.
  3. Produce an overall confidence score for the recommendation set.

This is the "generation" step of RAG — the retrieved song documents are
injected into the prompt context and Claude uses them to formulate its answer.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai

from catalog import song_to_document

logger = logging.getLogger(__name__)

_model: Optional[genai.GenerativeModel] = None


def _get_model() -> genai.GenerativeModel:
    global _model
    if _model is None:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        _model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=_SYSTEM_PROMPT,
        )
    return _model


_SYSTEM_PROMPT = """You are a music recommendation engine that has already retrieved
a shortlist of candidate songs for a user. Your job is to re-rank them and explain
why each one fits the user's request.

You will receive:
  - The user's original query (natural language).
  - A JSON list of candidate songs with their audio features and descriptions.

You must return ONLY a valid JSON array (no markdown, no preamble) of up to 5 objects,
sorted best-first, each with exactly these keys:
  {
    "id": <song id as string>,
    "title": <song title>,
    "artist": <artist name>,
    "explanation": <1-2 sentence plain-English reason this song fits the query>,
    "match_score": <float 0.0-1.0 — how well this song matches the query>,
    "confidence": <float 0.0-1.0 — how confident you are in this placement>
  }

Be specific in explanations — reference the user's actual words.
Return only the songs that genuinely fit. If fewer than 5 fit well, return fewer.
"""


def _build_candidate_context(candidates: List[Tuple[Dict, float]]) -> str:
    """Serialize retrieved candidates into the prompt context."""
    items = []
    for song, retrieval_score in candidates:
        items.append({
            "id": song["id"],
            "title": song["title"],
            "artist": song["artist"],
            "genre": song["genre"],
            "mood": song["mood"],
            "energy": song["energy"],
            "tempo_bpm": song["tempo_bpm"],
            "valence": song["valence"],
            "danceability": song["danceability"],
            "acousticness": song["acousticness"],
            "description": song_to_document(song),
            "retrieval_score": retrieval_score,
        })
    return json.dumps(items, indent=2)


def rank_and_explain(
    user_query: str,
    candidates: List[Tuple[Dict, float]],
    k: int = 5,
) -> List[Dict]:
    """
    Re-rank retrieved candidates using Claude and return enriched result dicts.

    Each result dict contains:
        id, title, artist, explanation, match_score, confidence
        + the original song data merged in.

    Returns an empty list on API failure (logged).
    """
    if not candidates:
        logger.warning("rank_and_explain called with empty candidates")
        return []

    candidate_context = _build_candidate_context(candidates)

    user_message = (
        f"User query: \"{user_query}\"\n\n"
        f"Candidate songs:\n{candidate_context}\n\n"
        f"Return the top {k} best matches as a JSON array."
    )

    logger.info("Calling Gemini to re-rank %d candidates for query: %r", len(candidates), user_query)
    model = _get_model()

    try:
        response = model.generate_content(user_message)
        raw = response.text.strip()
        logger.debug("Raw rank response: %s", raw)

        # Strip accidental markdown fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

        ranked = json.loads(raw)

        # Merge original song data back in
        song_lookup = {str(s["id"]): s for s, _ in candidates}
        for item in ranked:
            original = song_lookup.get(str(item["id"]), {})
            item["_song"] = original   # attach full song for downstream use

        logger.info(
            "Re-ranked %d songs. Top: %s (score=%.2f, confidence=%.2f)",
            len(ranked),
            ranked[0]["title"] if ranked else "—",
            ranked[0].get("match_score", 0) if ranked else 0,
            ranked[0].get("confidence", 0) if ranked else 0,
        )
        return ranked

    except json.JSONDecodeError as exc:
        logger.error("JSON parse failed in ranker: %s | raw=%r", exc, raw)
        return []
    except Exception as exc:
        logger.error("Gemini API error during ranking: %s", exc)
        return []
