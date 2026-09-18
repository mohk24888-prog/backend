from __future__ import annotations

import logging
import sys
from typing import Optional

from loguru import logger


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.remove()
    logger.add(sys.stdout, level=level, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}")
    return logger


def log_request(request) -> None:
    logger.info(
        "request: {} {} | client={}",
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )
