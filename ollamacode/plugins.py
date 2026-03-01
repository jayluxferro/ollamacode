"""
Plugin framework with event bus for OllamaCode.

Provides a simple event-driven plugin system:
  - EventBus: publish/subscribe event dispatcher
  - Plugin: base class with on_load()/on_unload() lifecycle hooks
  - PluginManager: loads plugins from a directory, manages lifecycle, emits events

Predefined events:
  tool_start, tool_end, message_sent, message_received,
  session_start, session_end, error

Usage in config (ollamacode.yaml):
  plugins_dir: ~/.ollamacode/plugins   # directory with .py plugin files
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Predefined event names (plugins may also use custom events)
TOOL_START = "tool_start"
TOOL_END = "tool_end"
MESSAGE_SENT = "message_sent"
MESSAGE_RECEIVED = "message_received"
SESSION_START = "session_start"
SESSION_END = "session_end"
ERROR = "error"


class EventBus:
    """Simple synchronous publish/subscribe event bus."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[..., Any]]] = {}

    def on(self, event: str, handler: Callable[..., Any]) -> None:
        """Register a handler for an event."""
        self._handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: Callable[..., Any]) -> None:
        """Remove a handler for an event."""
        handlers = self._handlers.get(event)
        if handlers:
            try:
                handlers.remove(handler)
            except ValueError:
                pass

    def emit(self, event: str, **kwargs: Any) -> None:
        """Emit an event, calling all registered handlers.

        Handlers are called synchronously in registration order.
        Exceptions are logged but do not stop other handlers.
        """
        for handler in self._handlers.get(event, []):
            try:
                handler(**kwargs)
            except Exception as exc:
                logger.debug(
                    "Plugin event handler %s raised on %r: %s",
                    getattr(handler, "__name__", handler),
                    event,
                    exc,
                )

    def clear(self, event: str | None = None) -> None:
        """Remove all handlers, or all handlers for a specific event."""
        if event is None:
            self._handlers.clear()
        else:
            self._handlers.pop(event, None)


class Plugin:
    """Base class for OllamaCode plugins.

    Subclass and override on_load() to register event handlers via self.bus.
    Override on_unload() for cleanup.
    """

    name: str = ""
    description: str = ""

    def __init__(self) -> None:
        self.bus: EventBus | None = None

    def on_load(self, bus: EventBus) -> None:
        """Called when the plugin is loaded. Register event handlers here."""
        self.bus = bus

    def on_unload(self) -> None:
        """Called when the plugin is unloaded. Clean up resources here."""
        pass


class PluginManager:
    """Manages plugin lifecycle and event emission."""

    def __init__(self) -> None:
        self.bus = EventBus()
        self._plugins: list[Plugin] = []

    def register(self, plugin: Plugin) -> None:
        """Register and load a single plugin instance."""
        try:
            plugin.on_load(self.bus)
            self._plugins.append(plugin)
            logger.debug("Plugin loaded: %s", plugin.name or plugin.__class__.__name__)
        except Exception as exc:
            logger.warning(
                "Failed to load plugin %s: %s",
                plugin.name or plugin.__class__.__name__,
                exc,
            )

    def unregister(self, plugin: Plugin) -> None:
        """Unload and remove a plugin."""
        try:
            plugin.on_unload()
        except Exception as exc:
            logger.debug(
                "Plugin %s raised on unload: %s",
                plugin.name or plugin.__class__.__name__,
                exc,
            )
        try:
            self._plugins.remove(plugin)
        except ValueError:
            pass

    def load_plugins_from_dir(self, path: str | Path) -> int:
        """Load all .py files in a directory as plugins.

        Each file should define a subclass of Plugin. The first Plugin subclass
        found in each module is instantiated and registered.

        Returns the number of plugins successfully loaded.
        """
        plugins_dir = Path(path).resolve()
        if not plugins_dir.is_dir():
            logger.debug("Plugins directory does not exist: %s", plugins_dir)
            return 0

        loaded = 0
        for py_file in sorted(plugins_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                mod_name = f"ollamacode_plugin_{py_file.stem}"
                spec = importlib.util.spec_from_file_location(mod_name, py_file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module
                spec.loader.exec_module(module)

                # Find the first Plugin subclass in the module
                plugin_cls = None
                for attr_name in dir(module):
                    obj = getattr(module, attr_name)
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, Plugin)
                        and obj is not Plugin
                    ):
                        plugin_cls = obj
                        break

                if plugin_cls is not None:
                    instance = plugin_cls()
                    if not instance.name:
                        instance.name = py_file.stem
                    self.register(instance)
                    loaded += 1
                else:
                    logger.debug("No Plugin subclass found in %s", py_file)
            except Exception as exc:
                logger.warning("Failed to load plugin file %s: %s", py_file, exc)

        return loaded

    def emit_event(self, event: str, **kwargs: Any) -> None:
        """Emit an event to all registered plugin handlers."""
        self.bus.emit(event, **kwargs)

    def unload_all(self) -> None:
        """Unload all registered plugins."""
        for plugin in list(self._plugins):
            self.unregister(plugin)

    @property
    def plugins(self) -> list[Plugin]:
        """Return a copy of the loaded plugins list."""
        return list(self._plugins)
