"""Configuration management for mempy."""

import os
from pathlib import Path
from typing import Optional


def get_storage_path(user_path: Optional[str] = None) -> Path:
    """
    Get the storage path for mempy data.

    Priority order:
    1. User-provided path
    2. MEMPY_HOME environment variable
    3. Default: ~/.mempy/data

    Args:
        user_path: Optional user-provided path

    Returns:
        Path object pointing to the storage directory

    Examples:
        >>> # Default path
        >>> get_storage_path()
        PosixPath('/home/user/.mempy/data')

        >>> # With environment variable
        >>> os.environ['MEMPY_HOME'] = '/custom/path'
        >>> get_storage_path()
        PosixPath('/custom/path')

        >>> # With user path
        >>> get_storage_path('/my/path')
        PosixPath('/my/path')
    """
    if user_path:
        return Path(user_path)

    env_path = os.environ.get("MEMPY_HOME")
    if env_path:
        return Path(env_path)

    return Path.home() / ".mempy" / "data"


def ensure_storage_dir(path: Optional[Path] = None) -> Path:
    """
    Ensure the storage directory exists, creating it if necessary.

    Args:
        path: Optional path to check/create. Uses get_storage_path() if None.

    Returns:
        Path object pointing to the storage directory
    """
    if path is None:
        path = get_storage_path()

    path.mkdir(parents=True, exist_ok=True)
    return path
