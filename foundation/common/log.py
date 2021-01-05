"""Implementation of customized logging configs."""
from __future__ import absolute_import, division, print_function

import logging
import logging.config
import sys
from typing import Optional

__all__ = ["configure_logging"]


def configure_logging(
    name: Optional[str] = None,
    lvl: int = logging.INFO,
    file: Optional[str] = None,
    file_only: bool = False,
) -> None:
    """
    Configure logging.

    If coloredlogs is available, use the colored formatter to print messages.

    # simplified code:
    ```
    logging.basicConfig(level=level,
                        format='%(asctime)s %(pathname)s:%(lineno)s %(message)s',
                        handlers=[logging.FileHandler(file, mode='w'),
                                logging.StreamHandler()])
    ```

    Args:
        name (str, optional): The root module name of this logger. If None, set root logger.
        lvl (int, optional): Logging level, include 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG',
            'NOTSET'.
        file (str, optional): Path to log file. If specified, add an extra handler that can write
            message to file.
        file_only (str, optional): Whether only log to file. Only valid if ``file`` is provided.
    """
    if file_only and file is None:
        raise ValueError("file should be provided when file_only=True")

    try:
        import coloredlogs  # noqa: F401

        color = True
    except ImportError:
        color = False

    fmt = "%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s"
    datefmt = "%m-%d %H:%M:%S"

    formatters = {
        "plain": {
            "format": fmt,
            "datefmt": datefmt,
        }
    }
    if color:
        formatters.update(
            {
                "color": {
                    "()": "coloredlogs.ColoredFormatter",
                    "format": fmt,
                    "datefmt": datefmt,
                },
            }
        )

    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "level": lvl,
            "formatter": "color" if color else "plain",
            "stream": sys.stdout,
        }
    }
    if file is not None:
        handlers.update(
            {
                "file": {
                    "class": "logging.FileHandler",
                    "filename": file,
                    "level": lvl,
                    "formatter": "plain",
                }
            }
        )

    if file is None:
        logger_handler = ["console"]
    else:
        if file_only:
            logger_handler = ["file"]
        else:
            logger_handler = ["console", "file"]

    if name is not None:
        logger_config = {
            "loggers": {name: {"level": lvl, "propagate": False, "handlers": logger_handler}}
        }
    else:
        logger_config = {
            "root": {
                "level": lvl,
                "handlers": logger_handler,
            }
        }

    logging.config.dictConfig(
        dict(
            version=1,
            disable_existing_loggers=False,
            formatters=formatters,
            handlers=handlers,
            **logger_config,
        )
    )
