"""Theme definitions for the OllamaCode Textual TUI.

Provides a palette of dark themes mapping semantic color names to hex values.
Each theme dict contains keys for primary UI colors, text colors, background
variants, border variants, and diff highlight colors.

Ported from OpenCode's theme system (dark variants only).
"""

from __future__ import annotations

from typing import TypedDict


class Theme(TypedDict):
    """Typed dictionary describing a complete TUI color theme."""

    primary: str
    secondary: str
    accent: str
    error: str
    warning: str
    success: str
    info: str
    text: str
    text_muted: str
    background: str
    background_panel: str
    background_element: str
    border: str
    border_active: str
    border_subtle: str
    diff_added: str
    diff_removed: str


def _make_theme(
    *,
    primary: str,
    secondary: str,
    accent: str,
    bg: str,
    text: str,
    border: str,
) -> Theme:
    """Build a full :class:`Theme` from the six core colours.

    Derived colours (error, warning, success, info, muted text, panel/element
    backgrounds, active/subtle borders, diff highlights) are either constants
    or computed from the supplied core values.
    """
    return Theme(
        primary=primary,
        secondary=secondary,
        accent=accent,
        error="#f38ba8",
        warning="#fab387",
        success="#a6e3a1",
        info="#89b4fa",
        text=text,
        text_muted=border,  # muted text reuses the border shade
        background=bg,
        background_panel=_lighten_hex(bg, 0.03),
        background_element=_lighten_hex(bg, 0.06),
        border=border,
        border_active=primary,
        border_subtle=_darken_hex(border, 0.15),
        diff_added="#a6e3a1",
        diff_removed="#f38ba8",
    )


# ---------------------------------------------------------------------------
# Colour math helpers
# ---------------------------------------------------------------------------


def _clamp(v: float) -> int:
    return max(0, min(255, int(round(v))))


def _lighten_hex(hex_color: str, amount: float) -> str:
    """Return *hex_color* lightened by *amount* (0..1)."""
    r, g, b = _hex_to_rgb(hex_color)
    r = _clamp(r + (255 - r) * amount)
    g = _clamp(g + (255 - g) * amount)
    b = _clamp(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _darken_hex(hex_color: str, amount: float) -> str:
    """Return *hex_color* darkened by *amount* (0..1)."""
    r, g, b = _hex_to_rgb(hex_color)
    r = _clamp(r * (1 - amount))
    g = _clamp(g * (1 - amount))
    b = _clamp(b * (1 - amount))
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Parse ``#rrggbb`` into an ``(r, g, b)`` tuple."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# Theme catalogue
# ---------------------------------------------------------------------------

THEMES: dict[str, Theme] = {
    "opencode": _make_theme(
        primary="#fab283",
        secondary="#7dc4e4",
        accent="#a6da95",
        bg="#1e1e2e",
        text="#cdd6f4",
        border="#313244",
    ),
    "dracula": _make_theme(
        primary="#bd93f9",
        secondary="#ff79c6",
        accent="#8be9fd",
        bg="#282a36",
        text="#f8f8f2",
        border="#44475a",
    ),
    "catppuccin-mocha": _make_theme(
        primary="#cba6f7",
        secondary="#f5c2e7",
        accent="#94e2d5",
        bg="#1e1e2e",
        text="#cdd6f4",
        border="#313244",
    ),
    "tokyo-night": _make_theme(
        primary="#7aa2f7",
        secondary="#bb9af7",
        accent="#7dcfff",
        bg="#1a1b26",
        text="#a9b1d6",
        border="#3b4261",
    ),
    "nord": _make_theme(
        primary="#88c0d0",
        secondary="#81a1c1",
        accent="#a3be8c",
        bg="#2e3440",
        text="#d8dee9",
        border="#3b4252",
    ),
    "gruvbox": _make_theme(
        primary="#d79921",
        secondary="#689d6a",
        accent="#458588",
        bg="#282828",
        text="#ebdbb2",
        border="#3c3836",
    ),
    "solarized-dark": _make_theme(
        primary="#268bd2",
        secondary="#2aa198",
        accent="#859900",
        bg="#002b36",
        text="#839496",
        border="#073642",
    ),
    "one-dark": _make_theme(
        primary="#61afef",
        secondary="#c678dd",
        accent="#98c379",
        bg="#282c34",
        text="#abb2bf",
        border="#3e4452",
    ),
    "rose-pine": _make_theme(
        primary="#c4a7e7",
        secondary="#ebbcba",
        accent="#9ccfd8",
        bg="#191724",
        text="#e0def4",
        border="#26233a",
    ),
    "monokai": _make_theme(
        primary="#f92672",
        secondary="#ae81ff",
        accent="#a6e22e",
        bg="#272822",
        text="#f8f8f2",
        border="#3e3d32",
    ),
}

DEFAULT_THEME = "opencode"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_theme(name: str) -> Theme:
    """Return the :class:`Theme` for *name*.

    Falls back to the default ``opencode`` theme when *name* is unknown.
    """
    return THEMES.get(name, THEMES[DEFAULT_THEME])


def list_themes() -> list[str]:
    """Return a sorted list of available theme names."""
    return sorted(THEMES.keys())


def generate_css(theme: dict[str, str]) -> str:
    """Return Textual CSS variable definitions for *theme*.

    The returned string can be injected as the application's ``CSS`` class
    variable or composed with other stylesheets.  Variable names use the
    pattern ``$<key>`` matching the references in ``styles.tcss`` (e.g.
    ``$primary``, ``$background-panel``, ``$text-muted``).

    Example output::

        $primary: #fab283;
        $secondary: #7dc4e4;
        ...
    """
    lines: list[str] = []
    for key, value in theme.items():
        css_var = f"${key.replace('_', '-')}"
        lines.append(f"{css_var}: {value};")
    return "\n".join(lines)
