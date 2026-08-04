"""
Project-wide logging.

Every new module goes through `get_logger(__name__)` instead of `print`, so the
whole pipeline has one switchable output channel. The handler writes to stdout
(not stderr) on purpose: tqdm owns stderr, and keeping the two apart stops the
progress bars from being chopped up by log lines.
"""

import logging
import sys

_CONFIGURED = False
_DEFAULT_LEVEL = logging.INFO
_FORMAT = '%(message)s'


def configure(level: int = _DEFAULT_LEVEL, fmt: str = _FORMAT):
    """Install the single stdout handler. Idempotent."""
    global _CONFIGURED

    root = logging.getLogger('nca_pipeline')
    root.setLevel(level)
    root.propagate = False

    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
        _CONFIGURED = True

    return root


def get_logger(name: str) -> logging.Logger:
    """
    Logger for one module.

    `name` is normally `__name__`; it is re-parented under `nca_pipeline` so a
    single call to `configure()` controls the level of the whole pipeline.
    """
    configure()
    return logging.getLogger(f'nca_pipeline.{name}')
