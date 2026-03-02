"""ASCII-art logo widget for the OllamaCode Textual TUI.

Displays a centered, theme-colored OllamaCode banner built on top of
Textual's :class:`~textual.widgets.Static` widget.
"""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

LOGO_ART = r"""
   ____  _ _                        ____          _
  / __ \| | |                      / ___|___   __| | ___
 | |  | | | | __ _ _ __ ___   __ _| |   / _ \ / _` |/ _ \
 | |  | | | |/ _` | '_ ` _ \ / _` | |  | (_) | (_| |  __/
 | |__| | | | (_| | | | | | | (_| | |___\___/ \__,_|\___|
  \____/|_|_|\__,_|_| |_| |_|\__,_|\____|
""".strip("\n")


class Logo(Static):
    """Centered ASCII art logo rendered in the theme's primary colour.

    Parameters
    ----------
    color:
        A Rich colour string (e.g. a hex value like ``#fab283``) applied to
        the logo text.  Defaults to the ``opencode`` theme primary if not
        supplied.
    """

    DEFAULT_CSS = """
    Logo {
        width: 100%;
        content-align: center middle;
        text-align: center;
    }
    """

    def __init__(
        self,
        color: str = "#fab283",
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        self._color = color
        super().__init__(id=id, classes=classes)

    def render(self) -> Text:  # type: ignore[override]
        """Return the logo as a :class:`rich.text.Text` object."""
        txt = Text(LOGO_ART, justify="center")
        txt.stylize(f"bold {self._color}")
        return txt
