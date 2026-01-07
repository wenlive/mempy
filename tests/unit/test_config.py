"""Tests for configuration management."""

import os
from pathlib import Path

import pytest

from mempy.config import get_storage_path, ensure_storage_dir


class TestGetStoragePath:
    """Tests for get_storage_path function."""

    def test_default_path(self, monkeypatch):
        """Test default storage path."""
        # Remove MEMPY_HOME if set
        monkeypatch.delenv("MEMPY_HOME", raising=False)

        path = get_storage_path()
        assert path == Path.home() / ".mempy" / "data"

    def test_env_variable(self, monkeypatch):
        """Test storage path from environment variable."""
        monkeypatch.setenv("MEMPY_HOME", "/custom/path")
        path = get_storage_path()
        assert path == Path("/custom/path")

    def test_user_path_override(self, monkeypatch):
        """Test user-provided path overrides everything."""
        monkeypatch.setenv("MEMPY_HOME", "/env/path")
        path = get_storage_path("/user/path")
        assert path == Path("/user/path")

    def test_user_path_takes_priority_over_env(self, monkeypatch):
        """Test user path takes priority over env variable."""
        monkeypatch.setenv("MEMPY_HOME", "/env/path")
        path = get_storage_path(user_path="/user/path")
        assert path == Path("/user/path")


class TestEnsureStorageDir:
    """Tests for ensure_storage_dir function."""

    def test_creates_directory_if_not_exists(self, tmp_path):
        """Test directory is created if it doesn't exist."""
        new_dir = tmp_path / "new_storage"
        result = ensure_storage_dir(new_dir)
        assert result == new_dir
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_existing_directory(self, tmp_path):
        """Test existing directory is not modified."""
        result = ensure_storage_dir(tmp_path)
        assert result == tmp_path
        assert tmp_path.exists()
