"""
main.py — Entry point for the RAG Music Recommender.

Run:
    python src/main.py                  # demo mode (preset queries)
    python src/main.py --interactive    # interactive REPL
    python src/main.py --query "..."    # single query
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure src/ is on the path when run from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from catalog import load_songs
from logger_setup import setup_logging
from pipeline import print_result, run

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

DEMO_QUERIES = [
    "something to help me focus late at night, not too upbeat",
    "I want to hype myself up before a workout, give me something intense",
    "chill Sunday morning music, maybe with some acoustic guitar vibes",
    "dark and moody electronic music for a late night drive",
    "happy music I can dance to at a house party",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Music Recommender")
    parser.add_argument("--interactive", action="store_true", help="Run as interactive REPL")
    parser.add_argument("--query", type=str, help="Single query to run")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG console logging")
    args = parser.parse_args()

    setup_logging(log_dir=LOG_DIR, debug=args.debug)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    songs = load_songs(DATA_PATH)
    print(f"Loaded {len(songs)} songs from catalog.\n")

    if args.query:
        result = run(args.query, songs)
        print_result(result)

    elif args.interactive:
        print("🎵  RAG Music Recommender — type your request, or 'quit' to exit.\n")
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if query.lower() in {"quit", "exit", "q"}:
                print("Goodbye.")
                break
            if not query:
                continue
            result = run(query, songs)
            print_result(result)

    else:
        # Demo mode
        print("Running demo queries...\n")
        for query in DEMO_QUERIES:
            result = run(query, songs)
            print_result(result)


if __name__ == "__main__":
    main()
