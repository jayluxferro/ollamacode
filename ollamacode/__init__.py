"""OllamaCode: coding assistant using local Ollama models and MCP tools."""

from importlib.metadata import PackageNotFoundError, version

from .config import get_resolved_config, load_config, merge_config_with_env

try:
    __version__ = version("ollamacode")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "get_resolved_config",
    "load_config",
    "merge_config_with_env",
]
