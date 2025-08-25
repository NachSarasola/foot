"""Utility styling helpers for the Ush analytics pipeline."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Core color palette used across graphics
COLORS = {
    "blue": "#0033a0",
    "cyan": "#00bcd4",
    "navy": "#0a1929",
    "grass": "#126f32",
    "fog": "#e6edf3",
}


def set_ush_theme() -> None:
    """Apply a lightweight matplotlib theme aligned with the Ush palette."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["navy"],
        "axes.facecolor": COLORS["navy"],
        "axes.edgecolor": COLORS["fog"],
        "axes.labelcolor": COLORS["fog"],
        "xtick.color": COLORS["fog"],
        "ytick.color": COLORS["fog"],
        "text.color": COLORS["fog"],
        "font.size": 10,
    })


def add_grass_texture(ax: plt.Axes, alpha: float = 0.1) -> None:
    """Overlay pale horizontal bands to mimic pitch grass."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    stripe = (ymax - ymin) / 20.0
    for i in range(20):
        if i % 2 == 0:
            ax.axhspan(ymin + i * stripe, ymin + (i + 1) * stripe,
                       facecolor="white", alpha=alpha, zorder=0)
    ax.set_facecolor(COLORS["grass"])


def scale_sizes(values, base: float = 100, k: float = 10,
                min_size: float = 40, max_size: float = 1000):
    """Scale numeric values into a bounded marker size array."""
    arr = base + k * np.asarray(values, dtype=float)
    return np.clip(arr, min_size, max_size)


def label(ax: plt.Axes, x: float, y: float, text: str, **kwargs) -> None:
    """Convenience wrapper to place player labels."""
    ax.text(x, y, text, ha="center", va="center",
            color=COLORS["fog"], fontsize=9, zorder=5, **kwargs)


def curved_edge(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float,
                weight: float = 1, color_main: str = "black") -> None:
    """Draw a gently curved connection between two nodes."""
    patch = FancyArrowPatch((x1, y1), (x2, y2),
                            connectionstyle="arc3,rad=0.15",
                            arrowstyle="-",
                            linewidth=max(0.5, weight * 0.6),
                            color=color_main, alpha=0.6, zorder=2)
    ax.add_patch(patch)


def annotate_goals_scatter(ax: plt.Axes, df) -> None:
    """Mark goals on a scatter-based shot map."""
    goals = df[df.get('is_goal', 0) == 1]
    for _, r in goals.iterrows():
        ax.scatter(r['x'], r['y'], s=120, marker='*',
                   color='#ffdf00', edgecolors=COLORS['fog'],
                   linewidths=0.5, zorder=5)


def annotate_goals_on_xg(ax: plt.Axes, df, team: str) -> None:
    """Annotate goals for a team on the xG race chart."""
    teams_order = list(dict.fromkeys(df['team']))
    color_map = {teams_order[0]: COLORS['blue']}
    if len(teams_order) > 1:
        color_map[teams_order[1]] = COLORS['cyan']
    goals = df[(df['team'] == team) & (df.get('is_goal', 0) == 1)]
    cum = 0.0
    for _, r in goals.iterrows():
        cum += r.get('xg_val', 0)
        ax.scatter(r.get('minute', 0), cum, s=40,
                   color=color_map.get(team, COLORS['fog']),
                   edgecolors='white', linewidths=0.4, zorder=5)
