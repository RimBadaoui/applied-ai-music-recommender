"""
query_parser.py — Stage 1 of the RAG pipeline.

Sends the user's free-text query to Gemini and asks it to extract a structured
preference object (genre, mood, energy, valence, etc.).  The result is then
used by catalog.retrieve_candidates() to pull the most relevant songs before
the final ranking step.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, Optional

import google.generativeai as genai

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


_SYSTEM_PROMPT = """You are a music preference parser. Your job is to read a user's
natural-language music request and extract a structured JSON object with the following keys:

  genre        string  (e.g. "lofi", "pop", "rock", "jazz", "ambient", "synthwave", "indie pop") or null
  mood         string  (e.g. "happy", "chill", "intense", "moody", "relaxed", "focused") or null
  energy       float   0.0 = very calm, 1.0 = extremely energetic
  valence      float   0.0 = dark/sad, 1.0 = bright/uplifting
  danceability float   0.0 = not danceable, 1.0 = highly danceable
  acousticness float   0.0 = fully electronic, 1.0 = fully acoustic
  tempo_bpm    float   estimated BPM (e.g. 60-180) — guess from context clues
  confidence   float   0.0-1.0 — how confident you are in this parse

Rules:
- Return ONLY valid JSON. No markdown fences, no preamble.
- If the user gives no signal for a field, omit it from the JSON (do not guess).
- confidence reflects how clearly the user's intent was expressed.
- Available genres: lofi, pop, rock, jazz, ambient, synthwave, indie pop
- Available moods: happy, chill, intense, moody, relaxed, focused
"""


def parse_query(user_query: str) -> Dict:
    """
    Call Gemini to extract structured music preferences from a free-text query.

    Returns a dict like:
        {
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.35,
            "confidence": 0.88,
            ...
        }

    On failure, returns {"confidence": 0.0, "_error": "..."}.
    """
    logger.info("Parsing query: %r", user_query)
    model = _get_model()
    raw = ""

    try:
        response = model.generate_content(user_query)
        raw = response.text.strip()
        logger.debug("Raw parse response: %s", raw)

        # Strip accidental markdown fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

        prefs = json.loads(raw)
        prefs["_raw_query"] = user_query
        logger.info("Parsed prefs (confidence=%.2f): %s", prefs.get("confidence", 0), prefs)
        return prefs

    except json.JSONDecodeError as exc:
        logger.error("JSON parse failed: %s | raw=%r", exc, raw)
        return {"confidence": 0.0, "_error": f"JSON parse failed: {exc}", "_raw_query": user_query}
    except Exception as exc:
        logger.error("Gemini API error during query parse: %s", exc)
        return {"confidence": 0.0, "_error": str(exc), "_raw_query": user_query}
