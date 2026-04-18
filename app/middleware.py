"""
Session logging middleware.

Intercepts every HTTP request/response, assigns a session ID, and appends a
single JSONL record to logs/sessions/{session_id}.jsonl.

Session ID resolution order:
  1. URL path  — extracted from /sessions/{id}/ or /sessions/{id} (end-of-path)
  2. Request header  — X-Session-ID (client reuse for stateless call sequences)
  3. Response body  — session_id field in JSON (handles POST /manual/sessions creation)
  4. Fresh UUID  — generated for all other requests

The final session ID is echoed back in the X-Session-ID response header so
callers can pin subsequent requests to the same log file.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.session_context import generate_session_id, set_session_id
from app.session_logger import append

_SESSION_PATH_RE = re.compile(r"/sessions/([a-zA-Z0-9_-]{6,})(?:/|$)")
_MANUAL_SESSIONS_PREFIX = "/api/v1/manual/sessions"
_SENSITIVE_HEADERS = {"authorization", "x-api-key", "cookie", "set-cookie"}


def _pick_session_id(request: Request) -> str:
    m = _SESSION_PATH_RE.search(request.url.path)
    if m:
        return m.group(1)
    header = request.headers.get("x-session-id")
    if header:
        return header
    return generate_session_id()


def _parse_body(body_bytes: bytes, content_type: str) -> Any:
    if not body_bytes:
        return None
    if "multipart/form-data" in content_type:
        return "<multipart upload — binary content omitted>"
    try:
        return json.loads(body_bytes)
    except Exception:
        text = body_bytes.decode("utf-8", errors="replace")
        return text[:4000] if len(text) > 4000 else text


def _safe_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        k: ("[REDACTED]" if k.lower() in _SENSITIVE_HEADERS else v)
        for k, v in headers.items()
    }


class SessionLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request and its response to logs/sessions/{session_id}.jsonl."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if not request.url.path.startswith(_MANUAL_SESSIONS_PREFIX):
            return await call_next(request)

        session_id = _pick_session_id(request)
        set_session_id(session_id)

        body_bytes = await request.body()
        request_body = _parse_body(body_bytes, request.headers.get("content-type", ""))

        t0 = time.monotonic()
        response = await call_next(request)
        elapsed_ms = round((time.monotonic() - t0) * 1000)

        resp_chunks: list[bytes] = []
        async for chunk in response.body_iterator:
            resp_chunks.append(chunk)
        resp_bytes = b"".join(resp_chunks)

        response_body = _parse_body(resp_bytes, response.media_type or "")

        # For session-creation responses, key the log file on the real session_id
        # returned by the service rather than the ephemeral UUID we assigned above.
        if (
            isinstance(response_body, dict)
            and "session_id" in response_body
            and request.url.path.rstrip("/").endswith("/sessions")
        ):
            session_id = response_body["session_id"]

        try:
            append(
                session_id,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session_id": session_id,
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": dict(request.query_params),
                    "request_headers": _safe_headers(dict(request.headers)),
                    "request_body": request_body,
                    "status_code": response.status_code,
                    "elapsed_ms": elapsed_ms,
                    "response_body": response_body,
                },
            )
        except Exception:
            pass  # never let logging break the response

        headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() != "content-length"
        }
        headers["x-session-id"] = session_id

        return Response(
            content=resp_bytes,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )
