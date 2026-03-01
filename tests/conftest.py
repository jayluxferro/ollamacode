"""
Shared fixtures for OllamaCode tests.

Provides commonly-needed fixtures:
  - mock_fs: temp workspace with OLLAMACODE_FS_ROOT set
  - temp_config: temp config directory with OLLAMACODE_CONFIG_DIR set
  - mock_ollama: patched ollama module that never hits the network
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_fs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temp workspace directory and set OLLAMACODE_FS_ROOT to it.

    Also sets OLLAMACODE_SANDBOX_LEVEL to 'full' so writes are allowed by default.
    Tests that need a specific sandbox level can override after this fixture runs.

    Returns the workspace Path.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("OLLAMACODE_FS_ROOT", str(workspace))
    monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", "full")
    return workspace


@pytest.fixture()
def temp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temp config directory and point state/config paths at it.

    Patches:
      - ollamacode.state._STATE_PATH
      - ollamacode.state._STATE_LOCK_PATH
      - HOME-based paths won't collide with user state

    Returns the config directory Path.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    state_path = config_dir / "state.json"
    lock_path = config_dir / "state.lock"
    monkeypatch.setattr("ollamacode.state._STATE_PATH", state_path)
    monkeypatch.setattr("ollamacode.state._STATE_LOCK_PATH", lock_path)
    # Reset the in-memory cache so tests start clean
    monkeypatch.setattr("ollamacode.state._state_cache", None)
    monkeypatch.setattr("ollamacode.state._state_cache_mtime_ns", -1)
    return config_dir


@pytest.fixture()
def mock_ollama() -> MagicMock:
    """Return a mock ollama module with embed and chat stubs.

    The mock embed returns a single zero-vector embedding by default.
    The mock chat returns a simple text response.
    """
    ollama_mock = MagicMock()

    class FakeEmbedResult:
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]

        def get(self, key: str, default: Any = None) -> Any:
            if key == "embeddings":
                return self.embeddings
            return default

    ollama_mock.embed.return_value = FakeEmbedResult()

    class FakeChatResult:
        message = {"content": "Hello from mock."}

        def get(self, key: str, default: Any = None) -> Any:
            if key == "message":
                return self.message
            return default

    ollama_mock.chat.return_value = FakeChatResult()
    return ollama_mock
