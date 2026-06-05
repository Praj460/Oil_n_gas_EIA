# oil_gas_pipeline | config/logging_config.py
# Centralized logging configuration
# Sets up file + console logging for every module in the pipeline
# Usage: from config.logging_config import setup_logging
#        setup_logging()   ← call once at the top of any entry point

import logging
import logging.handlers
from pathlib import Path
from config.config import config


def setup_logging(
    log_level:    str  = None,
    log_to_file:  bool = True,
    log_filename: str  = "pipeline.log",
) -> None:
    """
    Configures logging for the entire pipeline.

    Sets up two handlers:
    - Console handler — INFO and above, clean format for terminal output
    - File handler    — DEBUG and above, detailed format with timestamps
                        rotates at 10MB, keeps last 5 files

    Call this once at the entry point of any script or DAG before
    any other imports so all loggers inherit the config.

    Args:
        log_level:    override log level e.g. "DEBUG" (defaults to config value)
        log_to_file:  if True, writes logs to logs/pipeline.log
        log_filename: name of the log file inside the logs/ directory
    """
    level_str = (log_level or config.env == "development" and "DEBUG" or "INFO").upper()
    level     = getattr(logging, level_str, logging.INFO)

    # ── Formatters ────────────────────────────────────────────────────────────

    # Console — clean, readable
    console_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # File — detailed, includes module path for debugging
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Root logger ───────────────────────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers — prevents duplicate logs if called twice
    root_logger.handlers.clear()

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # ── File handler ──────────────────────────────────────────────────────────
    if log_to_file:
        log_path = config.paths.logs / log_filename
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=10 * 1024 * 1024,   # 10 MB per file
            backupCount=5,               # keep last 5 rotated files
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_fmt)
        root_logger.addHandler(file_handler)

    # ── Silence noisy third-party loggers ────────────────────────────────────
    # These libraries log a lot at DEBUG level — suppress to WARNING
    for noisy_lib in [
        "urllib3",
        "requests",
        "httpx",
        "cmdstanpy",      # Prophet's backend
        "prophet",
        "numba",
        "matplotlib",
        "statsmodels",
        "sqlalchemy.engine",
    ]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    # ── Confirm setup ─────────────────────────────────────────────────────────
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured | "
        f"level={level_str} | "
        f"file={'logs/' + log_filename if log_to_file else 'disabled'}"
    )


# ── Module-level loggers for each pipeline component ─────────────────────────
# Import these in other files instead of calling logging.getLogger() each time
# e.g. from config.logging_config import ingestion_logger

ingestion_logger    = logging.getLogger("oil_gas.ingestion")
db_logger           = logging.getLogger("oil_gas.database")
transform_logger    = logging.getLogger("oil_gas.transform")
model_logger        = logging.getLogger("oil_gas.models")
quality_logger      = logging.getLogger("oil_gas.quality")
dashboard_logger    = logging.getLogger("oil_gas.dashboard")