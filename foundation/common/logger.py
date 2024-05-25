import atexit
import functools
import logging
import sys
from pathlib import Path

from coloredlogs import ColoredFormatter

__all__ = ["setup_logger"]


@functools.lru_cache
def setup_logger(
    name: str,
    level: int = logging.INFO,
    *,
    rank: int = 0,
    color: bool = True,
    output: str | None = None,
    file_mode: str = "w",
) -> logging.Logger:
    """Configures logger with given name.

    Args:
        name (str): Logger name.
        level (int): The logging level, include 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG',
            'NOTSET'. Default: logging.INFO
        rank (int): Rank of the current process. Default: 0 (main process)
        color (bool): Whether to use colorful formatter for main process.
        output (str, optional): Path to a filename or a directory. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name. Otherwise, logs will be saved
                to `output/log.txt`. Default: None
        file_mode (str): The file mode used to open `output`. Default: "w"
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    handlers = []

    fmt = "%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s"
    datefmt = "%m-%d %H:%M:%S"
    plain_formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # stdout handler: main process only
    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(ColoredFormatter(fmt=fmt, datefmt=datefmt) if color else plain_formatter)
        handlers.append(ch)

    # file logging: all workers
    if output is not None:
        output = Path(output)
        if output.suffix in (".txt", ".log"):
            filename = output.name
            parent = output.parent
        else:
            filename = "log.txt"
            parent = output

        parent.mkdir(parents=True, exist_ok=True)
        filename = filename + f".rank{rank}" if rank > 0 else filename
        output = parent / filename

        filestream_handler = logging.StreamHandler(_cached_log_stream(str(output), file_mode))
        filestream_handler.setFormatter(plain_formatter)
        handlers.append(filestream_handler)

    if len(handlers) == 0:
        logger.warning(
            "No handlers are added. Please considering set `rank` to 0 or specify `output`"
        )

    for handler in handlers:
        handler.setLevel(level)
        logger.addHandler(handler)

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.cache
def _cached_log_stream(filename: str, mode: str = "w"):
    # use 1K buffer if writing to cloud storage
    io = open(filename, mode, buffering=1024 if "://" in filename else -1)
    atexit.register(io.close)
    return io
