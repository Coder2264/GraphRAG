"""Application logging configuration."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv


load_dotenv()


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_LOG_LEVEL = "INFO"


def _resolve_log_level() -> int:
    """Read LOG_LEVEL from the environment, defaulting to INFO."""
    level_name = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    return getattr(logging, level_name, logging.INFO)


def setup_logging() -> None:
    """
    Ensure application logs are visible in the terminal.

    Uvicorn does not guarantee our package loggers inherit an INFO-level root,
    so we explicitly raise the root/app logger levels here. If no handlers
    exist, we attach a console handler.
    """
    level = _resolve_log_level()
    root_logger = logging.getLogger()
    app_logger = logging.getLogger("app")

    root_logger.setLevel(level)
    app_logger.setLevel(level)

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        root_logger.addHandler(handler)
