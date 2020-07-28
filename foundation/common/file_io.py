# Copyright (c) Open-MMLab. All rights reserved.
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import logging
from typing import IO, Any, List, Optional, Union

import six

from .file_handler import HandlerRegistry

__all__ = [
    'load',
    'dump',
    'str_from_file',
    'list_from_file',
]

logger = logging.getLogger(__name__)


def load(file: Union[str, IO], file_format: Optional[str] = None, **kwargs: Any) -> Any:
    """Loads data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    Args:
        file: Filename or a file-like object.
        file_format: If not specified (i.e. None), the file format will be inferred from the file
            extension, otherwise use the specified one. Currently supported formats include 'json',
            'yaml/yml' and 'pickle/pkl'.

    Returns:
        The content from the file.
    """
    if file_format is None and not isinstance(file, six.string_types):
        raise ValueError('file_format must be specified since file is not a path')

    if isinstance(file, six.string_types) and file_format is None:
        file_format = file.split('.')[-1]

    if not HandlerRegistry.contains(file_format):
        raise TypeError('Unsupported format: {}'.format(file_format))

    handler = HandlerRegistry.get(file_format)()
    if isinstance(file, six.string_types):
        obj = handler.load_from_filepath(file, **kwargs)
    elif hasattr(file, 'read'):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('file must be a path str or a file-object')
    return obj


def dump(
    obj: Any,
    file: Optional[Union[str, IO]] = None,
    file_format: Optional[str] = None,
    **kwargs: Any
) -> Optional[str]:
    """Dumps data to json/yaml/pickle strings or files.

    This method provides a unified api for dumping data as strings or to files, and also
    supports custom arguments for each file format.

    Args:
        obj: The python object to be dumped.
        file: If not specified, then the object is dump to a str, otherwise to a file specified by
            the filename or file-like object.
        file_format: Same as :func:`load`.
    """
    if file_format is None and not isinstance(file, six.string_types):
        raise ValueError('file_format must be specified since file is not a path')

    if isinstance(file, six.string_types) and file_format is None:
        file_format = file.split('.')[-1]

    if not HandlerRegistry.contains(file_format):
        raise TypeError('Unsupported format: {}'.format(file_format))

    handler = HandlerRegistry.get(file_format)()
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif isinstance(file, six.string_types):
        handler.dump_to_filepath(obj, file, **kwargs)
    elif hasattr(file, 'write'):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('file must be None, a path str or a file-object')


def str_from_file(filepath: str, n: int = -1) -> str:
    """Loads a text file and parses the content as a string.

    Args:
        filepath: Path to the file.
        n: Number of bytes to be read from the file. Read to the end of the file by default.

    Returns:
        The parsed contents.
    """
    with open(filepath, 'r') as f:
        return f.read(n)


def list_from_file(filepath: str, prefix: str = '', offset: int = 0, max_num: int = 0) -> List[str]:
    """Loads a text file and parses the content as a list of strings.

    Args:
        filepath: Path to the file.
        prefix: The prefix to be inserted to the beginning of each item.
        offset: The offset of lines. Default: 0
        max_num: The maximum number of lines to be read, less than or equal to zero means no
            limitation. Default: 0

    Returns:
        item_list: Contents of file.
    """
    cnt = 0
    item_list = []
    with open(filepath, 'r') as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip('\n'))
            cnt += 1
    return item_list
