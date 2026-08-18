"""Dependency-free exploratory data analysis helpers for NeuralForge Part 008."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def _values(values: Sequence[float], *, name: str = "values") -> tuple[float, ...]:
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one value")
    try:
        data = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if not all(math.isfinite(value) for value in data):
        raise ValueError(f"{name} must contain only finite values")
    return data


def quantile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated quantile using sorted sample positions."""

    data = sorted(_values(values))
    p = float(probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if len(data) == 1:
        return data[0]

    position = p * (len(data) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return data[lower]
    fraction = position - lower
    return data[lower] * (1.0 - fraction) + data[upper] * fraction


@dataclass(frozen=True, slots=True)
class NumericSummary:
    count: int
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    mean: float
    standard_deviation: float
    iqr: float


def describe(values: Sequence[float]) -> NumericSummary:
    data = _values(values)
    count = len(data)
    center = math.fsum(data) / count
    variance = math.fsum((value - center) ** 2 for value in data) / count
    q1 = quantile(data, 0.25)
    q3 = quantile(data, 0.75)
    return NumericSummary(
        count=count,
        minimum=min(data),
        q1=q1,
        median=quantile(data, 0.5),
        q3=q3,
        maximum=max(data),
        mean=center,
        standard_deviation=math.sqrt(variance),
        iqr=q3 - q1,
    )


def iqr_outlier_mask(values: Sequence[float], *, multiplier: float = 1.5) -> tuple[bool, ...]:
    data = _values(values)
    factor = float(multiplier)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("multiplier must be finite and greater than zero")
    q1 = quantile(data, 0.25)
    q3 = quantile(data, 0.75)
    spread = q3 - q1
    lower = q1 - factor * spread
    upper = q3 + factor * spread
    return tuple(value < lower or value > upper for value in data)


@dataclass(frozen=True, slots=True)
class Histogram:
    edges: tuple[float, ...]
    counts: tuple[int, ...]

    @property
    def total(self) -> int:
        return sum(self.counts)


def histogram(values: Sequence[float], *, bins: int = 10) -> Histogram:
    data = _values(values)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")

    low = min(data)
    high = max(data)
    if low == high:
        half_width = 0.5 if low == 0.0 else abs(low) * 0.05
        if half_width == 0.0:
            half_width = 0.5
        low -= half_width
        high += half_width

    width = (high - low) / bins
    edges = tuple(low + width * index for index in range(bins + 1))
    counts = [0] * bins

    for value in data:
        if value == high:
            index = bins - 1
        else:
            index = int((value - low) / width)
            index = min(max(index, 0), bins - 1)
        counts[index] += 1

    return Histogram(edges=edges, counts=tuple(counts))


def scatter_svg(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    title: str = "Scatter Plot",
    width: int = 720,
    height: int = 480,
) -> str:
    """Create a small self-contained SVG scatter plot without plotting libraries."""

    x = _values(x_values, name="x values")
    y = _values(y_values, name="y values")
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of values")
    if width < 200 or height < 200:
        raise ValueError("width and height must each be at least 200 pixels")

    padding = 60.0
    plot_width = width - 2.0 * padding
    plot_height = height - 2.0 * padding

    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    if min_x == max_x:
        min_x -= 0.5
        max_x += 0.5
    if min_y == max_y:
        min_y -= 0.5
        max_y += 0.5

    def map_x(value: float) -> float:
        return padding + (value - min_x) / (max_x - min_x) * plot_width

    def map_y(value: float) -> float:
        return height - padding - (value - min_y) / (max_y - min_y) * plot_height

    circles = "\n".join(
        f'  <circle cx="{map_x(a):.2f}" cy="{map_y(b):.2f}" r="4" />'
        for a, b in zip(x, y)
    )
    safe_title = html.escape(title, quote=True)

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>{safe_title}</title>
  <rect x="0" y="0" width="{width}" height="{height}" fill="white" />
  <text x="{width / 2:.1f}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{safe_title}</text>
  <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="black" />
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="black" />
{circles}
</svg>'''


def write_scatter_svg(
    path: str | Path,
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    title: str = "Scatter Plot",
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        scatter_svg(x_values, y_values, title=title),
        encoding="utf-8",
    )
    return destination
