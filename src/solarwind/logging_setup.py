"""Structured logging setup shared by every command line entry point."""

import sys

from loguru import logger

LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} | {message} | {extra}"


def configure_logging(level: str = "INFO") -> None:
    """Replace the default loguru handler with one stderr handler at the given level."""
    logger.remove()
    logger.add(sys.stderr, level=level, format=LOG_FORMAT)
