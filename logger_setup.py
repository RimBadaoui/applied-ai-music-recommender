"""
logger_setup.py — Configures structured logging for the entire application.

Writes two handlers:
  - Console: human-readable (WARNING+ only to keep output clean)
  - File:    JSON-structured (DEBUG+) for full traceability
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(log_dir: str = "logs", debug: bool = False) -> None:
    """
    Call once at application startup.

    Args:
        log_dir: Directory where the JSON log file will be written.
        debug:   If True, console handler also shows DEBUG messages.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "recommender.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # --- Console handler (human readable) ---
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if debug else logging.WARNING)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(console)

    # --- File handler (JSON structured) ---
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    logging.getLogger(__name__).info("Logging initialised. Log file: %s", log_path)
