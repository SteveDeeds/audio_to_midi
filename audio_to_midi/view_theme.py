"""Shared display-only colors for the CQT and standard notation."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ViewTheme:
    name: str
    background: str
    ink: str
    muted: str
    staff_line: str
    grid: str
    beat_grid: str
    bar_line: str
    cursor: str
    selection: str
    envelope: str
    overview: str
    authored: str
    generated: str
    note_edge: str
    selected_edge: str
    lyric: str
    accent: str
    row_overlay: tuple[int, int, int, int]
    pitch_line: tuple[int, int, int, int]
    cqt_colors: tuple[str, ...]
    label: str
    is_light: bool = False
    cqt_interpolation: str = "rgb"


THEMES = {
    "dark": ViewTheme(
        "dark", "#171d28", "#e2e8f0", "#8994a7", "#b5bfce", "#4c607a", "#b09a82",
        "#ffffff", "#4ade80", "#4fc3d9", "#42657c", "#101620", "#84dba7", "#6b9fe8",
        "#0d1118", "#ffffff", "#101820", "#f7b554", (0, 0, 0, 45), (150, 165, 195, 60),
        ("#0c101a", "#281e5a", "#962d6e", "#f07832", "#fcf0a0"), "Classic",
    ),
    "light": ViewTheme(
        "light", "#ffffff", "#1d2733", "#5d6877", "#7b8794", "#b7c2ce", "#8c7561",
        "#334155", "#137a3b", "#007f99", "#547487", "#f3efe5", "#217a46", "#2663aa",
        "#f4f6f8", "#15253b", "#ffffff", "#97610c", (80, 93, 109, 20), (70, 86, 110, 70),
        ("#f3efe5", "#d3e17c", "#23b823", "#0065c3", "#000000"), "Saved gradient", True,
    ),
}

for name, label, light_colors, dark_colors in (
    ("blue", "Blue to indigo",
     ("#f6f8f8", "#80a1ce", "#1d32a7", "#090762", "#060024"),
     ("#070809", "#2d4c75", "#4159de", "#807df6", "#bfb2ff")),
    ("mint", "Mint to teal",
     ("#f7f8f6", "#80ce91", "#1da772", "#076257", "#002124"),
     ("#070907", "#2d753c", "#41dea2", "#7df6e8", "#b2f9ff")),
    ("amber", "Amber to crimson",
     ("#f8f8f6", "#ceaa80", "#a7451d", "#621207", "#240000"),
     ("#090807", "#75542d", "#de6f41", "#f68c7d", "#ffb2b2")),
):
    for appearance, colors in (("light", light_colors), ("dark", dark_colors)):
        key = f"{appearance}-{name}"
        THEMES[key] = replace(
            THEMES[appearance], name=key, label=label, cqt_colors=colors, cqt_interpolation="hsl",
        )


def view_theme(name: str) -> ViewTheme:
    if name not in THEMES:
        raise ValueError(f"Unknown display theme: {name!r}.")
    return THEMES[name]
