"""
catalog.py — Loads songs.csv and builds a lightweight semantic index.

Each song is embedded as a plain-text "document" that describes its features
in natural language. At query time we use the Anthropic API to embed the
user's query AND each song document, then rank by cosine similarity (RAG step).
Because the Anthropic API does not expose an embeddings endpoint, we instead
use Claude to produce a structured similarity score for each song against the
query — this is the RAG retrieval step that feeds the recommendation pipeline.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

Song = Dict[str, object]   # plain dict; keys match CSV columns


def load_songs(csv_path: str) -> List[Song]:
    """Load songs.csv, cast numeric columns, return list of dicts."""
    numeric = {"energy", "tempo_bpm", "valence", "danceability", "acousticness"}
    songs: List[Song] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            for col in numeric:
                if col in row:
                    row[col] = float(row[col])
            songs.append(row)
    logger.info("Loaded %d songs from %s", len(songs), csv_path)
    return songs


# ---------------------------------------------------------------------------
# Song → natural-language document (the "corpus" for RAG)
# ---------------------------------------------------------------------------

def song_to_document(song: Song) -> str:
    """
    Convert a song dict into a descriptive natural-language string.
    This is what gets semantically compared to the user's free-text query.
    """
    energy_word = (
        "very high energy" if song["energy"] >= 0.8
        else "high energy" if song["energy"] >= 0.65
        else "moderate energy" if song["energy"] >= 0.45
        else "low energy"
    )
    acoustic_word = "very acoustic" if song["acousticness"] >= 0.7 else (
        "somewhat acoustic" if song["acousticness"] >= 0.4 else "electronic/produced"
    )
    dance_word = "highly danceable" if song["danceability"] >= 0.75 else (
        "moderately danceable" if song["danceability"] >= 0.5 else "not very danceable"
    )
    valence_word = "very positive/uplifting" if song["valence"] >= 0.75 else (
        "neutral" if song["valence"] >= 0.5 else "melancholic/dark"
    )

    return (
        f"'{song['title']}' by {song['artist']}. "
        f"Genre: {song['genre']}. Mood: {song['mood']}. "
        f"This track is {energy_word} at {song['tempo_bpm']} BPM. "
        f"It is {acoustic_word}, {dance_word}, and sounds {valence_word}."
    )


# ---------------------------------------------------------------------------
# Cosine similarity over manual feature vectors (no external embedding API)
# ---------------------------------------------------------------------------

def _feature_vector(song: Song) -> List[float]:
    """
    Return a simple numeric feature vector for a song.
    Used as a deterministic fallback similarity metric.
    """
    return [
        float(song["energy"]),
        float(song["valence"]),
        float(song["danceability"]),
        float(song["acousticness"]),
        float(song["tempo_bpm"]) / 200.0,   # normalise BPM to ~[0,1]
    ]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def retrieve_candidates(
    query_prefs: Dict,
    songs: List[Song],
    top_k: int = 8,
) -> List[Tuple[Song, float]]:
    """
    Stage-1 RAG retrieval: rank songs by cosine similarity to the
    structured preference vector extracted from the user query.

    query_prefs keys expected:
        energy        float  [0,1]
        valence       float  [0,1]   (optional, defaults to 0.5)
        danceability  float  [0,1]   (optional)
        acousticness  float  [0,1]   (optional)
        tempo_bpm     float          (optional)
    """
    q_vec = [
        query_prefs.get("energy", 0.5),
        query_prefs.get("valence", 0.5),
        query_prefs.get("danceability", 0.5),
        query_prefs.get("acousticness", 0.3),
        query_prefs.get("tempo_bpm", 100.0) / 200.0,
    ]

    scored: List[Tuple[Song, float]] = []
    for song in songs:
        s_vec = _feature_vector(song)
        sim = _cosine(q_vec, s_vec)

        # Genre/mood bonus so textual matches aren't ignored
        if query_prefs.get("genre") and song["genre"] == query_prefs["genre"]:
            sim += 0.15
        if query_prefs.get("mood") and song["mood"] == query_prefs["mood"]:
            sim += 0.10

        scored.append((song, round(sim, 4)))

    scored.sort(key=lambda x: x[1], reverse=True)
    logger.debug("Top-%d candidates: %s", top_k, [s["title"] for s, _ in scored[:top_k]])
    return scored[:top_k]
