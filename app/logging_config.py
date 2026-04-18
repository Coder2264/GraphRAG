"""Application logging configuration."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


APP_LOGGER_NAME = "app"
LLM_LOGGER_NAME = "app.llm"
CONSOLE_HANDLER_NAME = "graphrag-console"
FILE_HANDLER_NAME = "graphrag-file"
LLM_FILE_HANDLER_NAME = "graphrag-llm-file"
DEFAULT_LOG_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s"
DEFAULT_LLM_LOG_FORMAT = "[%(asctime)s]\n%(message)s\n"
DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE = "logs/app.log"
DEFAULT_LLM_LOG_FILE = "logs/LLM.log"


def _resolve_log_level() -> int:
    """Read LOG_LEVEL from the environment, defaulting to INFO."""
    level_name = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    return getattr(logging, level_name, logging.INFO)


def _resolve_log_file() -> Path:
    """Read LOG_FILE from the environment, defaulting to logs/app.log."""
    log_file = Path(os.getenv("LOG_FILE", DEFAULT_LOG_FILE)).expanduser()
    if log_file.is_absolute():
        return log_file
    return Path.cwd() / log_file


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_LOG_DATE_FORMAT)


def _get_named_handler(logger: logging.Logger, handler_name: str) -> logging.Handler | None:
    for handler in logger.handlers:
        if handler.get_name() == handler_name:
            return handler
    return None


def _configure_console_handler(logger: logging.Logger, level: int) -> None:
    handler = _get_named_handler(logger, CONSOLE_HANDLER_NAME)
    if handler is None:
        handler = logging.StreamHandler()
        handler.set_name(CONSOLE_HANDLER_NAME)
        logger.addHandler(handler)
    handler.setLevel(level)
    handler.setFormatter(_build_formatter())


def _configure_file_handler(logger: logging.Logger, level: int) -> None:
    handler = _get_named_handler(logger, FILE_HANDLER_NAME)
    log_file = _resolve_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if handler is None:
        handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        handler.set_name(FILE_HANDLER_NAME)
        logger.addHandler(handler)
    handler.setLevel(level)
    handler.setFormatter(_build_formatter())


def _resolve_llm_log_file() -> Path:
    """Read LLM_LOG_FILE from the environment, defaulting to logs/LLM.log."""
    log_file = Path(os.getenv("LLM_LOG_FILE", DEFAULT_LLM_LOG_FILE)).expanduser()
    if log_file.is_absolute():
        return log_file
    return Path.cwd() / log_file


def _configure_llm_file_handler(logger: logging.Logger, level: int) -> None:
    handler = _get_named_handler(logger, LLM_FILE_HANDLER_NAME)
    llm_log_file = _resolve_llm_log_file()
    llm_log_file.parent.mkdir(parents=True, exist_ok=True)

    if handler is None:
        handler = logging.FileHandler(llm_log_file, mode="a", encoding="utf-8")
        handler.set_name(LLM_FILE_HANDLER_NAME)
        logger.addHandler(handler)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(DEFAULT_LLM_LOG_FORMAT, datefmt=DEFAULT_LOG_DATE_FORMAT)
    )


def setup_logging() -> None:
    """
    Ensure application logs are visible in the terminal and persisted to disk.

    All loggers under the `app.*` namespace inherit from the `app` logger, so
    we configure handlers once there and keep propagation disabled to avoid
    duplicate entries from Uvicorn/root handlers.

    The `app.llm` sub-logger is isolated (propagate=False) so that full LLM
    prompts and responses go only to logs/LLM.log and do not flood app.log or
    the console.
    """
    level = _resolve_log_level()
    app_logger = logging.getLogger(APP_LOGGER_NAME)

    app_logger.setLevel(level)
    app_logger.propagate = False

    _configure_console_handler(app_logger, level)
    _configure_file_handler(app_logger, level)

    # LLM call logger — full prompts/responses go only to LLM.log
    llm_logger = logging.getLogger(LLM_LOGGER_NAME)
    llm_logger.setLevel(level)
    llm_logger.propagate = False
    _configure_llm_file_handler(llm_logger, level)
