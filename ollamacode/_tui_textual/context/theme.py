"""Theme definitions for the OllamaCode Textual TUI.

Provides 25 themes (dark and light variants) with semantic colour tokens.
Supports auto-detection of terminal background via OSC 11 query.

Ported from OpenCode's theme system with additional themes.
"""

from __future__ import annotations

import logging
import sys
from typing import TypedDict

logger = logging.getLogger(__name__)


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
    diff_added_bg: str
    diff_removed: str
    diff_removed_bg: str


# ---------------------------------------------------------------------------
# Colour math helpers
# ---------------------------------------------------------------------------


def _clamp(v: float) -> int:
    return max(0, min(255, int(round(v))))


def _lighten_hex(hex_color: str, amount: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    r = _clamp(r + (255 - r) * amount)
    g = _clamp(g + (255 - g) * amount)
    b = _clamp(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _darken_hex(hex_color: str, amount: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    r = _clamp(r * (1 - amount))
    g = _clamp(g * (1 - amount))
    b = _clamp(b * (1 - amount))
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _luminance(hex_color: str) -> float:
    """Calculate relative luminance of a hex colour (0..1)."""
    r, g, b = _hex_to_rgb(hex_color)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


# ---------------------------------------------------------------------------
# Theme builder
# ---------------------------------------------------------------------------


def _make_dark_theme(
    *,
    primary: str,
    secondary: str,
    accent: str,
    bg: str,
    text: str,
    border: str,
    error: str = "#f38ba8",
    warning: str = "#fab387",
    success: str = "#a6e3a1",
    info: str = "#89b4fa",
) -> Theme:
    """Build a dark :class:`Theme` from core colours."""
    return Theme(
        primary=primary,
        secondary=secondary,
        accent=accent,
        error=error,
        warning=warning,
        success=success,
        info=info,
        text=text,
        text_muted=border,
        background=bg,
        background_panel=_lighten_hex(bg, 0.03),
        background_element=_lighten_hex(bg, 0.06),
        border=border,
        border_active=primary,
        border_subtle=_darken_hex(border, 0.15),
        diff_added="#a6e3a1",
        diff_added_bg=_darken_hex("#a6e3a1", 0.75),
        diff_removed="#f38ba8",
        diff_removed_bg=_darken_hex("#f38ba8", 0.75),
    )


def _make_light_theme(
    *,
    primary: str,
    secondary: str,
    accent: str,
    bg: str,
    text: str,
    border: str,
    error: str = "#d20f39",
    warning: str = "#df8e1d",
    success: str = "#40a02b",
    info: str = "#1e66f5",
) -> Theme:
    """Build a light :class:`Theme` from core colours."""
    return Theme(
        primary=primary,
        secondary=secondary,
        accent=accent,
        error=error,
        warning=warning,
        success=success,
        info=info,
        text=text,
        text_muted=_lighten_hex(text, 0.35),
        background=bg,
        background_panel=_darken_hex(bg, 0.03),
        background_element=_darken_hex(bg, 0.06),
        border=border,
        border_active=primary,
        border_subtle=_lighten_hex(border, 0.15),
        diff_added="#40a02b",
        diff_added_bg=_lighten_hex("#40a02b", 0.85),
        diff_removed="#d20f39",
        diff_removed_bg=_lighten_hex("#d20f39", 0.85),
    )


# ---------------------------------------------------------------------------
# Theme catalogue — 25 themes
# ---------------------------------------------------------------------------

THEMES: dict[str, Theme] = {
    # ── Dark Themes ──
    "opencode": _make_dark_theme(
        primary="#fab283",
        secondary="#7dc4e4",
        accent="#a6da95",
        bg="#1e1e2e",
        text="#cdd6f4",
        border="#313244",
    ),
    "dracula": _make_dark_theme(
        primary="#bd93f9",
        secondary="#ff79c6",
        accent="#8be9fd",
        bg="#282a36",
        text="#f8f8f2",
        border="#44475a",
    ),
    "catppuccin-mocha": _make_dark_theme(
        primary="#cba6f7",
        secondary="#f5c2e7",
        accent="#94e2d5",
        bg="#1e1e2e",
        text="#cdd6f4",
        border="#313244",
    ),
    "tokyo-night": _make_dark_theme(
        primary="#7aa2f7",
        secondary="#bb9af7",
        accent="#7dcfff",
        bg="#1a1b26",
        text="#a9b1d6",
        border="#3b4261",
    ),
    "nord": _make_dark_theme(
        primary="#88c0d0",
        secondary="#81a1c1",
        accent="#a3be8c",
        bg="#2e3440",
        text="#d8dee9",
        border="#3b4252",
    ),
    "gruvbox": _make_dark_theme(
        primary="#d79921",
        secondary="#689d6a",
        accent="#458588",
        bg="#282828",
        text="#ebdbb2",
        border="#3c3836",
    ),
    "solarized-dark": _make_dark_theme(
        primary="#268bd2",
        secondary="#2aa198",
        accent="#859900",
        bg="#002b36",
        text="#839496",
        border="#073642",
    ),
    "one-dark": _make_dark_theme(
        primary="#61afef",
        secondary="#c678dd",
        accent="#98c379",
        bg="#282c34",
        text="#abb2bf",
        border="#3e4452",
    ),
    "rose-pine": _make_dark_theme(
        primary="#c4a7e7",
        secondary="#ebbcba",
        accent="#9ccfd8",
        bg="#191724",
        text="#e0def4",
        border="#26233a",
    ),
    "monokai": _make_dark_theme(
        primary="#f92672",
        secondary="#ae81ff",
        accent="#a6e22e",
        bg="#272822",
        text="#f8f8f2",
        border="#3e3d32",
    ),
    "aura": _make_dark_theme(
        primary="#a277ff",
        secondary="#82e2ff",
        accent="#61ffca",
        bg="#15141b",
        text="#edecee",
        border="#29263c",
    ),
    "ayu": _make_dark_theme(
        primary="#ffb454",
        secondary="#73d0ff",
        accent="#bae67e",
        bg="#0b0e14",
        text="#bfbdb6",
        border="#1c1f26",
    ),
    "everforest": _make_dark_theme(
        primary="#a7c080",
        secondary="#7fbbb3",
        accent="#d699b6",
        bg="#2d353b",
        text="#d3c6aa",
        border="#3d484d",
    ),
    "kanagawa": _make_dark_theme(
        primary="#dca561",
        secondary="#7e9cd8",
        accent="#98bb6c",
        bg="#1f1f28",
        text="#dcd7ba",
        border="#2a2a37",
    ),
    "matrix": _make_dark_theme(
        primary="#00ff41",
        secondary="#008f11",
        accent="#00ff41",
        bg="#0d0208",
        text="#00ff41",
        border="#003b00",
        success="#00ff41",
        info="#008f11",
        error="#ff0000",
        warning="#ffff00",
    ),
    "flexoki": _make_dark_theme(
        primary="#d0a215",
        secondary="#6f7bb6",
        accent="#879a39",
        bg="#100f0f",
        text="#cecdc3",
        border="#282726",
    ),
    "vercel": _make_dark_theme(
        primary="#ffffff",
        secondary="#888888",
        accent="#0070f3",
        bg="#000000",
        text="#ededed",
        border="#333333",
    ),
    "zenburn": _make_dark_theme(
        primary="#f0dfaf",
        secondary="#8cd0d3",
        accent="#7f9f7f",
        bg="#3f3f3f",
        text="#dcdccc",
        border="#4f4f4f",
    ),
    "carbonfox": _make_dark_theme(
        primary="#78a9ff",
        secondary="#be95ff",
        accent="#42be65",
        bg="#161616",
        text="#f2f4f8",
        border="#353535",
    ),
    "github-dark": _make_dark_theme(
        primary="#58a6ff",
        secondary="#bc8cff",
        accent="#3fb950",
        bg="#0d1117",
        text="#c9d1d9",
        border="#21262d",
    ),
    # ── Light Themes ──
    "catppuccin-latte": _make_light_theme(
        primary="#8839ef",
        secondary="#ea76cb",
        accent="#179299",
        bg="#eff1f5",
        text="#4c4f69",
        border="#ccd0da",
    ),
    "solarized-light": _make_light_theme(
        primary="#268bd2",
        secondary="#2aa198",
        accent="#859900",
        bg="#fdf6e3",
        text="#657b83",
        border="#eee8d5",
    ),
    "one-light": _make_light_theme(
        primary="#4078f2",
        secondary="#a626a4",
        accent="#50a14f",
        bg="#fafafa",
        text="#383a42",
        border="#e5e5e6",
    ),
    "github-light": _make_light_theme(
        primary="#0969da",
        secondary="#8250df",
        accent="#1a7f37",
        bg="#ffffff",
        text="#24292f",
        border="#d0d7de",
    ),
    "rose-pine-dawn": _make_light_theme(
        primary="#907aa9",
        secondary="#d7827e",
        accent="#56949f",
        bg="#faf4ed",
        text="#575279",
        border="#dfdad9",
    ),
}

DEFAULT_THEME = "opencode"


# ---------------------------------------------------------------------------
# Terminal background detection (OSC 11)
# ---------------------------------------------------------------------------


def detect_terminal_mode() -> str:
    """Detect whether the terminal has a dark or light background.

    Uses the OSC 11 escape sequence to query the terminal background colour.
    Returns ``"dark"`` or ``"light"``. Falls back to ``"dark"`` if detection
    fails (most terminals are dark).
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return "dark"

    try:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            # Send OSC 11 query
            sys.stdout.write("\033]11;?\033\\")
            sys.stdout.flush()

            # Wait up to 100ms for response
            if select.select([sys.stdin], [], [], 0.1)[0]:
                response = ""
                while select.select([sys.stdin], [], [], 0.05)[0]:
                    response += sys.stdin.read(1)

                # Parse response: ESC ] 11 ; rgb:RRRR/GGGG/BBBB ESC \
                if "rgb:" in response:
                    rgb_part = response.split("rgb:")[1].split("\033")[0]
                    parts = rgb_part.strip().split("/")
                    if len(parts) == 3:
                        r = int(parts[0][:2], 16)
                        g = int(parts[1][:2], 16)
                        b = int(parts[2][:2], 16)
                        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
                        return "light" if lum > 0.5 else "dark"
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except Exception:
        logger.debug("Terminal background detection failed", exc_info=True)

    return "dark"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_theme(name: str) -> Theme:
    """Return the :class:`Theme` for *name*.

    Falls back to the default ``opencode`` theme when *name* is unknown.
    If *name* is ``"auto"``, detects terminal mode and picks a matching theme.
    """
    if name == "auto":
        mode = detect_terminal_mode()
        return THEMES["catppuccin-latte" if mode == "light" else "opencode"]

    return THEMES.get(name, THEMES[DEFAULT_THEME])


def list_themes() -> list[str]:
    """Return a sorted list of available theme names."""
    return sorted(THEMES.keys())


def get_dark_themes() -> list[str]:
    """Return sorted list of dark theme names."""
    return [
        name for name in sorted(THEMES) if _luminance(THEMES[name]["background"]) < 0.25
    ]


def get_light_themes() -> list[str]:
    """Return sorted list of light theme names."""
    return [
        name
        for name in sorted(THEMES)
        if _luminance(THEMES[name]["background"]) >= 0.25
    ]


def generate_css(theme: dict[str, str]) -> str:
    """Return Textual CSS variable definitions for *theme*.

    The returned string can be injected as the application's ``CSS`` class
    variable or composed with other stylesheets.  Variable names use the
    pattern ``$<key>`` matching the references in ``styles.tcss``.
    """
    lines: list[str] = []
    for key, value in theme.items():
        css_var = f"${key.replace('_', '-')}"
        lines.append(f"{css_var}: {value};")
    return "\n".join(lines)
