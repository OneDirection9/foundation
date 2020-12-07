"""Implementation of customized logging configs."""
from __future__ import absolute_import, division, print_function

import logging
import logging.config
from typing import Optional

__all__ = ["configure_logging"]


def configure_logging(
    level: int = logging.INFO, file: Optional[str] = None, mode: str = "w", root_mode: int = 1
) -> None:
    """Configures logging.

    # simplified code:
    ```
    logging.basicConfig(level=level,
                        format='%(asctime)s %(pathname)s:%(lineno)s %(message)s',
                        handlers=[logging.FileHandler(file, mode='w'),
                                logging.StreamHandler()])
    ```

    Args:
        level: Logging level, include 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'.
        file: Path to log file. If specified, add an extra handler that can write message to file.
        mode: Specify the mode in which the logging file is opened. Default: `w`
        root_mode: 0: both console and file logging; 1: console logging only; 2: file logging only.
            Default: 1.
    """
    if root_mode in (0, 2) and file is None:
        raise ValueError("file should be specified when root_handler_type is 0 or 2")

    format = "%(asctime)s %(filename)s:%(lineno)d[%(process)d] %(levelname)s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S.%f"

    basic_formatters = {
        "basic": {
            "format": format,
            "datefmt": datefmt,
        },
        "colored": {
            "()": "coloredlogs.ColoredFormatter",
            "format": format,
            "datefmt": datefmt,
        },
    }

    basic_handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": "colored",
        }
    }

    if file is not None:
        extra_handlers = {
            "file": {
                "class": "logging.FileHandler",
                "filename": file,
                "mode": mode,
                "level": level,
                "formatter": "basic",
            }
        }
    else:
        extra_handlers = {}

    if root_mode == 0:
        root_handlers = ["console", "file"]
    elif root_mode == 1:
        root_handlers = ["console"]
    elif root_mode == 2:
        root_handlers = ["file"]
    else:
        raise ValueError("root_mode can only be 0, 1, 2, but got {}".format(root_mode))

    logging.config.dictConfig(
        dict(
            version=1,
            disable_existing_loggers=False,
            formatters=basic_formatters,
            handlers=dict(**basic_handlers, **extra_handlers),
            root={
                "level": level,
                "handlers": root_handlers,
            },
        )
    )
