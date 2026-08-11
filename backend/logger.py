"""
Logging configuration module for RAG application.
Provides structured and JSON logging for debugging, Cloud Run, and monitoring.
"""

import json
import logging
import os
import sys
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Production JSON log formatter for GCP Cloud Run / Datadog."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logger(name: str = "RAG", level: str = "INFO") -> logging.Logger:
    """Set up and configure logger for the application."""
    logger = logging.getLogger(name)
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    use_json = os.getenv("LOG_FORMAT", "text").lower() == "json"
    if use_json:
        console_handler.setFormatter(JSONFormatter())
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get existing logger or create new one."""
    return logging.getLogger(name)
