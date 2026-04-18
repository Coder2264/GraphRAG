"""
Request-scoped session ID context variable.

Provides a ContextVar that any layer in the call stack can read/write during
a single request without threading it through function signatures.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

_session_id_var: ContextVar[str] = ContextVar("session_id", default="")


def get_session_id() -> str:
    """Return the session ID bound to the current async context."""
    return _session_id_var.get()


def set_session_id(session_id: str) -> None:
    """Bind a session ID to the current async context."""
    _session_id_var.set(session_id)


def generate_session_id() -> str:
    """Generate a fresh 12-char hex session ID."""
    return uuid.uuid4().hex[:12]
