"""
Session file logger — appends JSONL entries to logs/sessions/{session_id}.jsonl.

Each line is one complete request/response record, making sessions fully
replayable by reading a single file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_SESSION_DIR = Path(os.getenv("SESSION_LOG_DIR", "logs/sessions"))


def append(session_id: str, entry: dict[str, Any]) -> None:
    """Append *entry* as a JSON line to the session's log file.

    Creates logs/sessions/ if it does not exist.  Silently does nothing when
    session_id is empty.
    """
    if not session_id:
        return
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = _SESSION_DIR / f"{session_id}.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
