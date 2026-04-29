# RAG Music Recommender

## Original project

**Music Recommender Simulation** (Modules 1–3) — a rule-based music recommendation system that accepts structured user preference dicts (`{"genre": "pop", "mood": "happy", "energy": 0.9}`) and scores each song in a CSV catalog using weighted feature matching (genre match +2.0, mood match +1.0, energy proximity 0–1). It returned ranked lists with brief score breakdowns. It had no AI, no natural-language understanding, and no ability to handle anything other than exact field values.

---

## Title and summary

**RAG Music Recommender** — a three-stage applied AI system that accepts *natural-language* music requests and returns semantically matched, confidence-scored, and explained song recommendations.

Instead of requiring the user to fill in a structured form, you can now say things like:

> *"something dark and moody for a late night drive"*
> *"I want to hype myself up before a workout"*
> *"chill Sunday morning with acoustic guitar vibes"*

Claude translates your request into structured preferences (Stage 1), retrieves the most relevant songs using cosine similarity (Stage 2), then re-ranks and explains every recommendation in plain English using the retrieved song documents as context (Stage 3). Guardrails validate each stage; every decision is logged to a JSON file.

---

## Architecture overview

```
User query (natural language)
        │
        ▼
[Guardrail] validate_query()
        │
        ▼
[Stage 1] query_parser.py
  Claude reads the query → returns structured JSON prefs
  (genre, mood, energy, valence, danceability, tempo_bpm, confidence)
        │
        ▼
[Guardrail] validate_parsed_prefs()
        │
        ▼
[Stage 2] catalog.py — retrieve_candidates()
  Cosine similarity between pref vector and each song's feature vector
  + genre/mood bonus → top-8 candidates returned
        │                              ▲
        │               songs.csv ─────┘
        ▼
[Stage 3] ranker.py — rank_and_explain()
  Retrieved song documents injected into Claude's prompt context (RAG)
  Claude re-ranks candidates and writes a natural-language explanation
  for each, plus match_score and confidence
        │
        ▼
[Guardrail] validate_results()
        │
        ▼
PipelineResult → printed to stdout
        │
        └──► logs/recommender.log (JSON, every stage)
```

**Components:**

| File | Role |
|------|------|
| `src/catalog.py` | Loads songs.csv, converts songs to NL documents, cosine similarity retrieval |
| `src/query_parser.py` | Stage 1 — Claude parses NL query into structured prefs dict |
| `src/ranker.py` | Stage 3 — Claude re-ranks retrieved candidates and writes explanations |
| `src/guardrails.py` | Validates inputs and outputs at every stage |
| `src/pipeline.py` | Orchestrates all three stages; returns `PipelineResult` |
| `src/logger_setup.py` | Dual-handler logging (console + JSON file) |
| `src/main.py` | Entry point: demo mode, interactive REPL, or single `--query` flag |
| `tests/test_pipeline.py` | 26 unit tests covering guardrails, catalog, and cosine math |
| `data/songs.csv` | 15-song catalog with energy, valence, tempo, genre, mood, etc. |

---

## Setup instructions

### 1. Clone / unzip the project

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Gemini API key

```bash
set GEMINI_API_KEY=YOUR_KEY
```

### 4. Run the tests (no API key needed)

```bash
python test_pipeline.py
# Expected: Ran 26 tests in ~0.01s — OK
```

### 5. Run in demo mode (5 preset queries)

```bash
python main.py --interactive
```

### 6. Run a single query

```bash
python main.py --query "something mellow and jazzy for a Sunday morning"
```

### 7. Run interactively

```bash
python src/main.py --interactive
```

### 8. Enable debug logging to console

```bash
python main.py --debug --query "intense workout music"
```

Full structured logs always go to `logs/recommender.log`.

---

## Sample interactions

### Query 1 — late night focus

**Input:**
```
something to help me focus late at night, not too upbeat
```

**Output:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Query   : something to help me focus late at night, not too upbeat
  Parsed  : genre=lofi, mood=focused, energy=0.35
  Parse confidence : 0.92
  Overall confidence: 0.88

  Top 3 recommendations:

  1. Focus Flow — LoRoom
     Match score : 0.96
     Confidence  : 0.95
     Why         : A lofi track with a focused mood at 80 BPM perfectly suits
                   your request for calm, late-night concentration music.

  2. Forest Path — Paper Lanterns
     Match score : 0.91
     Confidence  : 0.90
     Why         : Highly acoustic and focused in mood, this slow lofi track
                   matches the "not too upbeat" constraint closely.

  3. Library Rain — Paper Lanterns
     Match score : 0.87
     Confidence  : 0.83
     Why         : Chill mood and very low energy make this ideal for quiet
                   late-night studying.
```

---

### Query 2 — pre-workout hype

**Input:**
```
I want to hype myself up before a workout, give me something intense
```

**Output:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Query   : I want to hype myself up before a workout, give me something intense
  Parsed  : genre=None, mood=intense, energy=0.92
  Parse confidence : 0.95
  Overall confidence: 0.91

  Top 3 recommendations:

  1. Gym Hero — Max Pulse
     Match score : 0.97
     Confidence  : 0.96
     Why         : Extremely high energy (0.93) with an intense mood, this pop
                   track was practically designed for pre-workout hype.

  2. Storm Runner — Voltline
     Match score : 0.93
     Confidence  : 0.92
     Why         : Rock track at 152 BPM with intense mood — exactly the
                   adrenaline boost you asked for before hitting the gym.

  3. Neon Jungle — Voltline
     Match score : 0.89
     Confidence  : 0.88
     Why         : High energy rock with strong danceability to keep your
                   momentum going through a tough session.
```

---

### Query 3 — edge case, unknown genre

**Input:**
```
bossa nova vibes for a dinner party
```

**Output:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Query   : bossa nova vibes for a dinner party
  Parsed  : genre=bossa nova, mood=relaxed, energy=0.38
  Parse confidence : 0.72
  ⚠  Genre 'bossa nova' is not in the catalog. Known genres: ambient,
     indie pop, jazz, lofi, pop, rock, synthwave.

  Top 3 recommendations:

  1. Coffee Shop Stories — Slow Stereo
     Match score : 0.88
     Confidence  : 0.85
     Why         : Jazz is the closest available genre to bossa nova — relaxed
                   mood and acoustic warmth make this the best dinner party
                   fit in the catalog.

  2. Sunday Brunch — Slow Stereo
     Match score : 0.84
     Confidence  : 0.80
     Why         : Happy jazz at 95 BPM, highly acoustic, great for
                   background ambience at a social gathering.
```

The system warns the user that bossa nova isn't in the catalog but still recovers gracefully by finding the nearest equivalent.

---

### Query 4 — guardrail rejection

**Input:**
```
1234 5678
```

**Output:**
```
  ✗ Errors:
    • Query appears to contain no readable text.
```

---

## Design decisions

**Why cosine similarity instead of a vector embedding API?**
The Anthropic API does not expose an embeddings endpoint. Rather than add a second provider (OpenAI, Cohere), we use a deterministic feature vector — energy, valence, danceability, acousticness, normalised BPM — which are the actual audio features that matter for music matching. Genre and mood bonuses are added on top. This keeps the system self-contained and reproducible.

**Why two Claude calls per query?**
Stage 1 (parsing) and Stage 3 (ranking/explaining) have different tasks and different output schemas. Combining them into one prompt would require Claude to simultaneously parse the intent AND evaluate 8 candidate songs, creating a single large, fragile prompt. Keeping them separate gives cleaner JSON output from each stage and makes each step independently testable and replaceable.

**Why warn instead of hard-reject on unknown genres/moods?**
Unknown genres (like "bossa nova") indicate the user described something real but outside the current catalog. A hard rejection would be unhelpful. Warnings let the system attempt a best-effort result and tell the user exactly why it approximated.

**Trade-offs:**
- The 15-song catalog is too small for retrieval quality to vary much. With hundreds of songs, the cosine retrieval step becomes much more important.
- Two Claude API calls per query adds ~3–5 seconds of latency. For a production system, Stage 1 parsing results could be cached for repeated similar queries.
- The feature vector does not encode genre or mood numerically (they remain string matches), so genre-adjacent similarity ("bossa nova" ≈ "jazz") is handled by Claude in Stage 3, not by the retrieval stage.

---

## Testing summary

**Unit tests (26 tests, all passing):**

| Suite | Tests | What's covered |
|-------|-------|----------------|
| `TestValidateQuery` | 6 | Empty, whitespace, too short, too long, no letters, valid |
| `TestValidateParsedPrefs` | 6 | Parser errors, low confidence, out-of-range numeric, unknown genre/mood |
| `TestValidateResults` | 4 | Empty results, missing explanation, low confidence, valid |
| `TestCatalog` | 10 | Row count, type casting, energy range, NL doc generation, retrieval order, genre boost, cosine math |

**What worked:** Guardrails caught every class of bad input correctly. The cosine retrieval consistently surfaces the right genre-matching songs. Claude's JSON parsing is stable across all tested queries when given the strict system prompt.

**What didn't work initially:**
- Claude occasionally returned JSON wrapped in markdown fences despite being told not to. Fixed with a regex strip in both `query_parser.py` and `ranker.py`.
- Early versions of the query parser omitted fields rather than defaulting them, causing KeyErrors downstream. Fixed by using `.get()` with explicit defaults throughout `catalog.retrieve_candidates()`.

**What we learned:**
- Separating guardrails from business logic makes debugging much faster — you know immediately which stage failed.
- Structured JSON logging (one object per line) makes it easy to grep for specific events across a long session.

---

## Reflection

This project showed how a structured, deterministic system (scoring rules + CSV lookup) can be extended into a genuine AI pipeline without becoming opaque. The key insight is that RAG works best when retrieval and generation have clearly separated responsibilities: retrieval is fast, deterministic, and testable; generation (Claude) adds reasoning and language. Neither step alone would work as well.

The hardest part was not the code — it was prompt engineering. Getting Claude to return stable JSON from two different prompts, at two different abstraction levels, required careful system prompt design and defensive parsing on both ends. This mirrors real production AI systems, where prompt reliability is often the primary engineering challenge.

The guardrails layer also changed how the system *felt* to use. A system that explains what it can't do (and why) is much more trustworthy than one that silently fails or hallucinates. Even in a small demo, the warning messages for unknown genres made the system feel honest rather than pretending to know things it doesn't.
