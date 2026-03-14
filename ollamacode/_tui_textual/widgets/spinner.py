"""Knight Rider-style scanner spinner widget.

A bidirectional scanning animation with alpha-fading trail, inspired by
OpenCode's spinner. Falls back to a braille spinner on narrow terminals.
"""

from __future__ import annotations

from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _blend(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> str:
    """Blend two RGB colours by factor *t* (0 = c1, 1 = c2)."""
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return _rgb_to_hex(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def _derive_trail_colors(bright: str, bg: str, steps: int = 4) -> list[str]:
    """Derive a gradient of *steps* colours fading from *bright* to *bg*."""
    c_bright = _hex_to_rgb(bright)
    c_bg = _hex_to_rgb(bg)
    return [_blend(c_bright, c_bg, i / steps) for i in range(steps)]


def _derive_inactive(bright: str, bg: str, factor: float = 0.15) -> str:
    """Derive a very dim dot colour between *bright* and *bg*."""
    return _blend(_hex_to_rgb(bright), _hex_to_rgb(bg), 1 - factor)


class KnightRiderSpinner(Widget):
    """Bidirectional scanning spinner with alpha-fading trail.

    Parameters
    ----------
    label:
        Text shown to the right of the animation.
    color:
        Primary (bright) colour of the scanner head.
    bg_color:
        Background colour used for inactive dots and trail fading.
    width:
        Number of character positions in the scanner bar.
    trail_length:
        Number of trailing fade positions behind the head.
    """

    DEFAULT_CSS = """
    KnightRiderSpinner {
        width: auto;
        height: 1;
    }
    """

    label: reactive[str] = reactive("")

    # Block characters for the scanner
    HEAD_CHAR = "\u2588"  # full block
    TRAIL_CHAR = "\u2593"  # dark shade
    DIM_CHAR = "\u2591"  # light shade
    INACTIVE_CHAR = "\u00b7"  # middle dot

    def __init__(
        self,
        label: str = "",
        color: str = "#fab283",
        bg_color: str = "#1e1e2e",
        width: int = 12,
        trail_length: int = 4,
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self.label = label
        self._color = color
        self._bg_color = bg_color
        self._width = width
        self._trail_length = trail_length

        # Pre-compute trail colours
        self._trail_colors = _derive_trail_colors(color, bg_color, trail_length)
        self._inactive_color = _derive_inactive(color, bg_color)

        # Animation state
        self._pos = 0
        self._direction = 1  # 1 = right, -1 = left
        self._hold_start = 8  # frames to hold at start
        self._hold_end = 4  # frames to hold at end
        self._hold_counter = 0

    def on_mount(self) -> None:
        """Start the animation timer at ~15 fps."""
        self.set_interval(1 / 15, self._advance)

    def _advance(self) -> None:
        """Move the scanner head one step."""
        if self._hold_counter > 0:
            self._hold_counter -= 1
            self.refresh()
            return

        self._pos += self._direction
        if self._pos >= self._width - 1:
            self._pos = self._width - 1
            self._direction = -1
            self._hold_counter = self._hold_end
        elif self._pos <= 0:
            self._pos = 0
            self._direction = 1
            self._hold_counter = self._hold_start

        self.refresh()

    def render(self) -> Text:
        """Render the scanner bar with trailing fade."""
        result = Text()

        for i in range(self._width):
            distance = abs(self._pos - i)
            # Only show trail *behind* the head (in the opposite direction of travel)
            behind = (self._direction == 1 and i < self._pos) or (
                self._direction == -1 and i > self._pos
            )

            if distance == 0:
                # Scanner head
                result.append(self.HEAD_CHAR, style=f"bold {self._color}")
            elif behind and distance <= self._trail_length:
                # Trail positions — fade based on distance
                trail_idx = distance - 1
                if trail_idx < len(self._trail_colors):
                    color = self._trail_colors[trail_idx]
                    char = self.TRAIL_CHAR if distance <= 2 else self.DIM_CHAR
                    result.append(char, style=color)
                else:
                    result.append(self.INACTIVE_CHAR, style=self._inactive_color)
            else:
                # Inactive position
                result.append(self.INACTIVE_CHAR, style=self._inactive_color)

        if self.label:
            result.append(f" {self.label}", style="dim")

        return result


# Keep backward compat alias
BrailleSpinner = KnightRiderSpinner
