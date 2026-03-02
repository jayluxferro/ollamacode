"""Default keybinding definitions for the OllamaCode Textual TUI.

All bindings are expressed as Textual key-string literals and collected in
:data:`DEFAULT_KEYBINDS`.  Widget and screen code should read from this
mapping rather than hard-coding key strings so that users can override
bindings in one place in the future.
"""

from __future__ import annotations

# Mapping of logical action name -> Textual key string.
DEFAULT_KEYBINDS: dict[str, str] = {
    "submit": "enter",
    "newline": "shift+enter",
    "new_session": "ctrl+n",
    "clear": "ctrl+l",
    "command_palette": "ctrl+p",
    "toggle_sidebar": "ctrl+backslash",
    "cancel": "escape",
    "switch_agent": "tab",
    "quit": "ctrl+c",
    "scroll_up": "pageup",
    "scroll_down": "pagedown",
}
