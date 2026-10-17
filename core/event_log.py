"""Structured local event logging for diagnostics and performance analysis."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "audio_to_midi.jsonl"
_LOGGER = logging.getLogger("audio_to_midi.events")


def _configure() -> None:
    if _LOGGER.handlers:
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


def log_event(event: str, **fields: Any) -> None:
    _configure()
    payload = {"event": str(event), **fields}
    _LOGGER.info(json.dumps(payload, sort_keys=True, default=str))