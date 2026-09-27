"""Tiny presentation helpers shared by every lesson.

Pure-Python + numpy. No matplotlib required: we draw charts in ASCII so the
lessons run in any terminal, on any machine, with no display. If you *do* have
matplotlib installed you can use `maybe_show()` at the end of a lesson, but
nothing here needs it.
"""

from __future__ import annotations

import numpy as np


def title(text: str) -> None:
    """A section banner, so the printed narration is easy to scan."""
    bar = "═" * len(text)
    print(f"\n{bar}\n{text}\n{bar}")


def step(text: str) -> None:
    print(f"\n▸ {text}")


def ascii_scatter(
    xs: np.ndarray,
    ys: np.ndarray,
    labels: np.ndarray,
    width: int = 56,
    height: int = 18,
    xlabel: str = "x",
    ylabel: str = "y",
) -> None:
    """Scatter two classes in the terminal. '#' = malicious (1), '.' = benign (0)."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    grid = [[" "] * width for _ in range(height)]

    def cell(x: float, y: float) -> tuple[int, int]:
        xn = (x - xs.min()) / (np.ptp(xs) or 1.0)
        yn = (y - ys.min()) / (np.ptp(ys) or 1.0)
        col = min(width - 1, max(0, int(xn * (width - 1))))
        row = min(height - 1, max(0, int((1 - yn) * (height - 1))))
        return row, col

    for x, y, lab in zip(xs, ys, labels):
        r, c = cell(x, y)
        mark = "#" if lab >= 0.5 else "."
        # let '#' win a shared cell so attacks stay visible
        if grid[r][c] == " " or mark == "#":
            grid[r][c] = mark

    print(f"\n   {ylabel} ↑   ( # = malicious   . = benign )")
    for row in grid:
        print("      |" + "".join(row))
    print("      +" + "-" * width + f"→ {xlabel}")


def ascii_bars(names, values, width: int = 40) -> None:
    """Horizontal bar chart for things like feature weights."""
    values = [float(v) for v in values]
    lo = min(0.0, min(values))
    hi = max(0.0, max(values))
    span = (hi - lo) or 1.0
    zero_col = int(round((0 - lo) / span * width))
    for name, v in zip(names, values):
        col = int(round((v - lo) / span * width))
        line = [" "] * (width + 1)
        line[zero_col] = "|"
        if col >= zero_col:
            for i in range(zero_col + 1, col + 1):
                line[i] = "█"
        else:
            for i in range(col, zero_col):
                line[i] = "█"
        print(f"  {name:>16} {''.join(line)} {v:+.3f}")


def ascii_curve(f, lo: float, hi: float, width: int = 56, height: int = 12,
                label: str = "") -> None:
    """Plot a 1-D function f over [lo, hi]."""
    xs = np.linspace(lo, hi, width)
    ys = np.array([f(x) for x in xs], float)
    ylo, yhi = ys.min(), ys.max()
    span = (yhi - ylo) or 1.0
    grid = [[" "] * width for _ in range(height)]
    for c, y in enumerate(ys):
        r = min(height - 1, max(0, int((1 - (y - ylo) / span) * (height - 1))))
        grid[r][c] = "•"
    if label:
        print(f"\n  {label}")
    for row in grid:
        print("   " + "".join(row))
    print("   " + "-" * width)
    print(f"   x from {lo:g} to {hi:g}")


def maybe_show():
    """No-op unless you want to add matplotlib plots yourself; keeps lessons
    dependency-light. Returns True if matplotlib is available."""
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False
