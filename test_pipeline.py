"""
test_pipeline.py — Automated tests for the RAG Music Recommender.

Tests cover:
  - Guardrail validation (query, prefs, results)
  - Catalog loading and retrieval
  - Score sanity checks
  - Edge cases and bad inputs

Run with:  python -m pytest tests/test_pipeline.py -v
       or: python tests/test_pipeline.py
"""

from __future__ import annotations

import os
import sys
import unittest

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from catalog import load_songs, retrieve_candidates, song_to_document, _cosine
from guardrails import validate_parsed_prefs, validate_query, validate_results

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateQuery(unittest.TestCase):

    def test_valid_query(self):
        r = validate_query("something chill for studying")
        self.assertTrue(r.valid)
        self.assertEqual(r.errors, [])

    def test_empty_query(self):
        r = validate_query("")
        self.assertFalse(r.valid)
        self.assertTrue(any("empty" in e.lower() for e in r.errors))

    def test_whitespace_only(self):
        r = validate_query("   ")
        self.assertFalse(r.valid)

    def test_too_short(self):
        r = validate_query("ok")
        self.assertFalse(r.valid)

    def test_too_long(self):
        r = validate_query("x " * 300)
        self.assertFalse(r.valid)

    def test_no_letters(self):
        r = validate_query("1234 5678")
        self.assertFalse(r.valid)


class TestValidateParsedPrefs(unittest.TestCase):

    def test_valid_prefs(self):
        prefs = {"genre": "lofi", "mood": "chill", "energy": 0.4, "confidence": 0.9}
        r = validate_parsed_prefs(prefs)
        self.assertTrue(r.valid)

    def test_parser_error_flagged(self):
        prefs = {"_error": "JSON parse failed", "confidence": 0.0}
        r = validate_parsed_prefs(prefs)
        self.assertFalse(r.valid)

    def test_low_confidence_is_warning_not_error(self):
        prefs = {"energy": 0.5, "confidence": 0.1}
        r = validate_parsed_prefs(prefs)
        self.assertTrue(r.valid)        # still valid
        self.assertTrue(len(r.warnings) > 0)

    def test_energy_out_of_range_warning(self):
        prefs = {"energy": 1.5, "confidence": 0.8}
        r = validate_parsed_prefs(prefs)
        self.assertTrue(r.valid)        # warning, not error
        self.assertTrue(any("energy" in w.lower() for w in r.warnings))

    def test_unknown_genre_is_warning(self):
        prefs = {"genre": "bossa_nova", "confidence": 0.7}
        r = validate_parsed_prefs(prefs)
        self.assertTrue(r.valid)
        self.assertTrue(any("genre" in w.lower() for w in r.warnings))

    def test_unknown_mood_is_warning(self):
        prefs = {"mood": "euphoric", "confidence": 0.7}
        r = validate_parsed_prefs(prefs)
        self.assertTrue(r.valid)
        self.assertTrue(any("mood" in w.lower() for w in r.warnings))


class TestValidateResults(unittest.TestCase):

    def test_empty_results_invalid(self):
        r = validate_results([])
        self.assertFalse(r.valid)

    def test_valid_results(self):
        results = [
            {"title": "Song A", "explanation": "Great match.", "match_score": 0.9, "confidence": 0.85},
        ]
        r = validate_results(results)
        self.assertTrue(r.valid)

    def test_missing_explanation_is_warning(self):
        results = [{"title": "Song B", "explanation": "", "match_score": 0.7, "confidence": 0.6}]
        r = validate_results(results)
        self.assertTrue(r.valid)
        self.assertTrue(len(r.warnings) > 0)

    def test_low_confidence_is_warning(self):
        results = [{"title": "Song C", "explanation": "OK.", "match_score": 0.5, "confidence": 0.1}]
        r = validate_results(results)
        self.assertTrue(r.valid)
        self.assertTrue(any("confidence" in w.lower() for w in r.warnings))


# ─────────────────────────────────────────────────────────────────────────────
# Catalog tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCatalog(unittest.TestCase):

    def setUp(self):
        self.songs = load_songs(DATA_PATH)

    def test_loads_correct_count(self):
        self.assertEqual(len(self.songs), 15)

    def test_numeric_fields_cast(self):
        for song in self.songs:
            self.assertIsInstance(song["energy"], float)
            self.assertIsInstance(song["tempo_bpm"], float)
            self.assertIsInstance(song["valence"], float)

    def test_energy_in_range(self):
        for song in self.songs:
            self.assertGreaterEqual(song["energy"], 0.0)
            self.assertLessEqual(song["energy"], 1.0)

    def test_song_to_document_non_empty(self):
        for song in self.songs:
            doc = song_to_document(song)
            self.assertGreater(len(doc), 20)
            self.assertIn(song["title"], doc)
            self.assertIn(song["artist"], doc)

    def test_retrieve_returns_k(self):
        prefs = {"energy": 0.8, "valence": 0.8, "genre": "pop"}
        results = retrieve_candidates(prefs, self.songs, top_k=5)
        self.assertEqual(len(results), 5)

    def test_retrieve_scores_descending(self):
        prefs = {"energy": 0.5}
        results = retrieve_candidates(prefs, self.songs, top_k=10)
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_genre_match_boosts_score(self):
        """Songs matching the requested genre should rank higher on average."""
        prefs = {"energy": 0.4, "genre": "lofi"}
        results = retrieve_candidates(prefs, self.songs, top_k=5)
        top_genres = [s["genre"] for s, _ in results[:3]]
        self.assertIn("lofi", top_genres)

    def test_cosine_identical_vectors(self):
        v = [0.5, 0.5, 0.5]
        self.assertAlmostEqual(_cosine(v, v), 1.0, places=5)

    def test_cosine_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(_cosine(a, b), 0.0, places=5)

    def test_cosine_zero_vector(self):
        self.assertEqual(_cosine([0, 0, 0], [1, 2, 3]), 0.0)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
