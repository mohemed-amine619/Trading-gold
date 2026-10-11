"""
core/logging_setup.py — Centralised, cross-platform logging configuration.

Why this exists
---------------
The application previously had no central logging setup: each module created
its own logger and nothing created the ``logs/`` directory. On Windows a
missing log directory is a hard failure the moment a handler is attached, so
this module makes directory creation part of configuration.

Design rules:

* Timestamps are **UTC** internally. The mandate requires UTC storage; the
  trading engine must not depend on the machine's local timezone.
* Credentials are never logged. ``redact`` masks secrets before they reach a
  handler.
* Configuration is idempotent — calling it twice does not duplicate handlers.
* Works identically on Windows and POSIX; paths come from ``pathlib``.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Union

# NOTE: the stdlib has no module-level `logging.converter`. `Formatter.formatTime`
# uses `time.localtime` unless a `converter` argument is passed to the Formatter,
# which means the default is machine-local time. Timestamps for a trading engine
# must be UTC regardless of the host timezone, so we subclass and override.
class UTCFormatter(logging.Formatter):
    """Formatter that renders timestamps in UTC, never local time."""

    def formatTime(self, record, datefmt=None):  # noqa: N802 - stdlib name
        return time.strftime(datefmt or self.default_time_format,
                             time.gmtime(record.created))


DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
UTC_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

# Patterns that must never reach a log file.
_SECRET_PATTERNS = (
    # Keys may be prefixed (MT5_PASSWORD, db_password) or suffixed, so do not
    # anchor on a word boundary before the key name.
    re.compile(r"(?i)([\w.-]*(?:password|passwd|pwd)[\w.-]*)\s*[:=]\s*\S+"),
    re.compile(r"(?i)([\w.-]*(?:token|secret|api[_-]?key)[\w.-]*)\s*[:=]\s*\S+"),
    # Whole value after the header, e.g. "Authorization: Bearer abc123".
    re.compile(r"(?i)([\w.-]*authorization[\w.-]*)\s*[:=]\s*\S.*$"),
    re.compile(r"://([^:/@\s]+):([^@/\s]+)@"),  # credentials inside a URL
)


def redact(text: str) -> str:
    """
    Mask credentials in a string before it is logged.

    Applies to connection URLs and key=value secrets. Conservative by design:
    over-masking a log line is preferable to leaking a password.
    """
    if not text:
        return text
    result = text
    result = _SECRET_PATTERNS[3].sub(r"://\1:***@", result)
    for pattern in _SECRET_PATTERNS[:3]:
        result = pattern.sub(lambda m: f"{m.group(1)}=***", result)
    return result


class RedactingFilter(logging.Filter):
    """Logging filter that masks secrets in every emitted record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = redact(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {
                        k: redact(v) if isinstance(v, str) else v
                        for k, v in record.args.items()
                    }
                else:
                    record.args = tuple(
                        redact(a) if isinstance(a, str) else a for a in record.args
                    )
        except Exception:
            # Never let redaction break logging.
            pass
        return True


def resolve_log_dir(log_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve and create the log directory.

    Creating it here is what makes Windows safe: attaching a FileHandler to a
    missing directory raises immediately on every platform, but on Windows it
    also commonly happens because the directory was never shipped with git.
    """
    if log_dir is None:
        from core.settings import get_settings
        log_dir = get_settings().log_dir
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logging(
    level: Union[int, str] = logging.INFO,
    log_dir: Optional[Union[str, Path]] = None,
    *,
    to_file: bool = True,
    to_console: bool = True,
    log_filename: str = "app.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> Path:
    """
    Configure root logging. Idempotent.

    Returns the resolved log directory. UTC timestamps are used regardless of
    machine timezone.
    """
    root = logging.getLogger()
    root.setLevel(level if isinstance(level, int) else str(level).upper())

    # Idempotence: remove handlers this function previously installed.
    marker = "_trading_platform_handler"
    for handler in list(root.handlers):
        if getattr(handler, marker, False):
            root.removeHandler(handler)
            handler.close()

    formatter = UTCFormatter(DEFAULT_FORMAT, datefmt=UTC_FORMAT)
    redactor = RedactingFilter()

    if to_console:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(formatter)
        console.addFilter(redactor)
        setattr(console, marker, True)
        root.addHandler(console)

    directory = resolve_log_dir(log_dir) if to_file else Path(".")
    if to_file:
        # Rotating handler: a runaway log cannot fill the disk on a desktop PC.
        file_handler = logging.handlers.RotatingFileHandler(
            directory / log_filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redactor)
        setattr(file_handler, marker, True)
        root.addHandler(file_handler)

    # Keep SQLAlchemy and HTTP access logs out of INFO unless asked for.
    for noisy in ("sqlalchemy.engine", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return directory


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Module-level logger helper. Uses the configured root handlers."""
    return logging.getLogger(name or __name__)


# Reduce noise from libraries that log on import.
logging.getLogger("asyncio").setLevel(logging.WARNING)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")