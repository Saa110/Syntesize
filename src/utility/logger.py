"""
KnowAD Logger — Centralized logging for the entire project.

Usage:
    from neuro_symbolic_monitor.utils.logger import get_logger, setup_logging

    # In any module — get a child logger
    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.debug("Detailed info for log file only")

    # At the entry point of a script — setup handlers once
    setup_logging(log_dir="results", experiment_name="detection")

Log Levels:
    - DEBUG:   Detailed diagnostics (log file only)
    - INFO:    Progress, results, summaries (console + file)
    - WARNING: Recoverable issues, fallbacks
    - ERROR:   Failures that stop a component
    - CRITICAL: Unrecoverable failures

Output:
    - Console: INFO+ with colored, concise format
    - File:    DEBUG+ with full timestamps and module names
               Saved to <log_dir>/<experiment_name>_<timestamp>.log
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional


# ── Project root logger name ────────────────────────────────────────────────
ROOT_LOGGER_NAME = 'knowad'

# Track whether setup has been called
_is_configured = False


class _ColorFormatter(logging.Formatter):
    """Colored console formatter for readable terminal output."""

    COLORS = {
        logging.DEBUG:    '\033[90m',      # grey
        logging.INFO:     '\033[36m',      # cyan
        logging.WARNING:  '\033[33m',      # yellow
        logging.ERROR:    '\033[31m',      # red
        logging.CRITICAL: '\033[1;31m',    # bold red
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def __init__(self, fmt=None, datefmt=None):
        """
        Initialize the ColorFormatter.
        
        Args:
            fmt (str, optional): The log format string. Defaults to None.
            datefmt (str, optional): The date format string. Defaults to None.
            
        Returns:
            None
        """
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record):
        """
        Format the specified record with color if applicable.
        
        Args:
            record (logging.LogRecord): The log record to format.
            
        Returns:
            str: The formatted log string.
        """
        color = self.COLORS.get(record.levelno, '')
        record.levelname_colored = f"{color}{record.levelname:<7}{self.RESET}"

        # Shorten module names for console readability
        name = record.name
        if name.startswith(f'{ROOT_LOGGER_NAME}.'):
            name = name[len(ROOT_LOGGER_NAME) + 1:]
        record.short_name = name

        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name (str): Module name, typically ``__name__``. Will be prefixed
                    with 'knowad.' if not already.

    Returns:
        logging.Logger: A child logger under the 'knowad' root logger.

    Example::

        logger = get_logger(__name__)
        logger.info("Loading data...")
    """
    if not name.startswith(ROOT_LOGGER_NAME):
        # Convert module paths: neuro_symbolic_monitor.data.loader -> knowad.data.loader
        short = name.replace('neuro_symbolic_monitor.', '').replace('neuro_symbolic_monitor', '')
        if short.startswith('.'):
            short = short[1:]
        full_name = f"{ROOT_LOGGER_NAME}.{short}" if short else ROOT_LOGGER_NAME
    else:
        full_name = name

    return logging.getLogger(full_name)


def setup_logging(
    log_dir: str = 'results',
    experiment_name: str = 'experiment',
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    timestamp: Optional[str] = None,
) -> str:
    """
    Configure logging for the project — call once at script entry point.

    Sets up two handlers on the root 'knowad' logger:
    1. Console handler (INFO by default) — colored, concise
    2. File handler (DEBUG by default) — full detail with timestamps

    Args:
        log_dir (str, optional): Directory to write log files. Defaults to 'results'.
        experiment_name (str, optional): Prefix for the log filename. Defaults to 'experiment'.
        console_level (int, optional): Minimum level for console output. Defaults to logging.INFO.
        file_level (int, optional): Minimum level for file output. Defaults to logging.DEBUG.
        timestamp (Optional[str], optional): Optional timestamp string. Auto-generated if None.

    Returns:
        str: Path to the log file.
    """
    global _is_configured

    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")

    # Get the root project logger
    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)  # Allow all levels; handlers filter

    # Clear any existing handlers (safe for re-runs / notebooks)
    root.handlers.clear()

    # ── Console handler ──
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console_fmt = _ColorFormatter(
        fmt='%(asctime)s %(levelname_colored)s %(message)s',
        datefmt='%H:%M:%S'
    )
    console.setFormatter(console_fmt)
    root.addHandler(console)

    # ── File handler ──
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_fmt = logging.Formatter(
        fmt='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_fmt)
    root.addHandler(file_handler)

    # Prevent duplicate logs from propagating to the root Python logger
    root.propagate = False

    _is_configured = True

    root.info(f"Logging initialized → {log_file}")
    root.debug(f"Console level: {logging.getLevelName(console_level)}, "
               f"File level: {logging.getLevelName(file_level)}")

    return log_file


def is_configured() -> bool:
    """
    Check if setup_logging has been called.
    
    Args:
        None
        
    Returns:
        bool: True if logging is configured, False otherwise.
    """
    return _is_configured
