"""Braille-dot spinner widget for the OllamaCode Textual TUI.

Displays a compact animated spinner using braille characters, matching the
OpenCode spinner pattern.  An optional *label* is rendered beside the
spinning character.
"""

from __future__ import annotations

from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget


class BrailleSpinner(Widget):
    """Animated braille-dot spinner with an optional text label.

    The spinner cycles through the standard braille animation frames at
    10 fps and renders the current frame followed by the label text.

    Parameters
    ----------
    label:
        Text shown to the right of the spinning character.
    color:
        Rich colour string applied to the spinner character.
    """

    FRAMES: str = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
    """The ten braille animation frames."""

    DEFAULT_CSS = """
    BrailleSpinner {
        width: auto;
        height: 1;
    }
    """

    frame_index: reactive[int] = reactive(0)
    label: reactive[str] = reactive("")

    def __init__(
        self,
        label: str = "",
        color: str = "#fab283",
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self.label = label
        self._color = color

    def on_mount(self) -> None:
        """Start the frame-advance timer at ~10 fps."""
        self.set_interval(1 / 10, self._advance)

    def _advance(self) -> None:
        """Move to the next animation frame."""
        self.frame_index = (self.frame_index + 1) % len(self.FRAMES)

    def render(self) -> Text:  # type: ignore[override]
        """Return the current spinner frame and label as :class:`rich.text.Text`."""
        char = self.FRAMES[self.frame_index]
        spinner_text = Text(char, style=f"bold {self._color}")
        if self.label:
            spinner_text.append(f" {self.label}")
        return spinner_text
