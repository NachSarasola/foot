"""Utility functions and styling helpers for Ush Analytics plots.

This module centralises the colour palette and a set of helpers that are
shared between scripts.  The functions are lightweight wrappers built on
matplotlib and mplsoccer so they can be reused in notebooks or other
visualisation pipelines.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "navy": "#0A2540",   # primary background
    "blue": "#0072B2",   # cb-friendly blue
    "cyan": "#56B4E9",   # cb-friendly cyan
    "orange": "#E69F00", # cb-friendly orange
    "purple": "#CC79A7", # cb-friendly purple
    "fog": "#E6EEF6",    # light lines/text
    "grass": "#1B4332",  # pitch background
    "paper": "#FFFFFF",
    "ink": "#001018",
    "goal": "#FF3366",
}

# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def set_ush_theme():
    """Apply the default Ush Analytics matplotlib theme.

    The theme adjusts global ``matplotlib.rcParams`` so it only needs to be
    called once at the start of a script.  It sets dark backgrounds with
    light foreground elements using the colours defined in :data:`COLORS`.
    """
    plt.rcParams.update({
        "figure.facecolor": COLORS["navy"],
        "axes.facecolor": COLORS["navy"],
        "axes.edgecolor": COLORS["fog"],
        "axes.labelcolor": COLORS["fog"],
        "xtick.color": COLORS["fog"],
        "ytick.color": COLORS["fog"],
        "text.color": COLORS["fog"],
        "font.family": "sans-serif",
    })


def add_grass_texture(ax, step: float = 5.0, alpha: float = 0.12,
                      color: str | None = None) -> None:
    """Add subtle horizontal stripes to mimic a mowed grass texture.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing a pitch drawn with :class:`mplsoccer.Pitch`.
    step : float, default 5.0
        Distance between stripes in pitch units (StatsBomb convention).
    alpha : float, default 0.12
        Transparency of the stripes.
    color : str, optional
        Colour of the stripes.  Defaults to ``COLORS['fog']``.
    """
    stripe_color = color or COLORS["fog"]
    ymin, ymax = ax.get_ylim()
    for y in np.arange(ymin, ymax, step * 2):
        ax.axhspan(y, y + step, facecolor=stripe_color, alpha=alpha, zorder=0)


def scale_sizes(values, base: float = 100, k: float = 1.0,
                min_size: float | None = None,
                max_size: float | None = None) -> np.ndarray:
    """Scale numeric values into marker sizes for scatter plots.

    Parameters
    ----------
    values : array-like
        Input numeric values.
    base : float, default 100
        Base marker size to offset from.
    k : float, default 1.0
        Multiplicative factor applied to ``values``.
    min_size, max_size : float, optional
        Optional bounds for the output sizes.

    Returns
    -------
    numpy.ndarray
        The scaled sizes.
    """
    arr = np.asarray(values, dtype=float)
    sizes = base + k * arr
    if min_size is not None:
        sizes = np.maximum(min_size, sizes)
    if max_size is not None:
        sizes = np.minimum(max_size, sizes)
    return sizes


def label(ax, x: float, y: float, text: str, *, y_offset: float = 3.0,
          color: str | None = None, fontsize: int = 9, zorder: int = 5,
          **kwargs) -> None:
    """Place a label slightly above a point on the pitch.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the label.
    x, y : float
        Coordinates in StatsBomb units.
    text : str
        Text to display.
    y_offset : float, default 3.0
        Vertical offset to avoid overlapping the marker at ``(x, y)``.
    color : str, optional
        Text colour, defaults to ``COLORS['navy']``.
    fontsize : int, default 9
        Font size of the label text.
    zorder : int, default 5
        Matplotlib z-order for the text.
    **kwargs :
        Additional keyword arguments forwarded to ``Axes.text``.
    """
    ax.text(x, y - y_offset, text, ha="center", va="top",
            fontsize=fontsize, color=color or COLORS["navy"],
            zorder=zorder, **kwargs)


def curved_edge(ax, x_start: float, y_start: float, x_end: float, y_end: float,
                weight: float = 1.0, color_main: str | None = None,
                color_bg: str | None = None, curvature: float = 0.2,
                alpha: float = 0.6, zorder: int = 2) -> None:
    """Draw a curved edge between two nodes on a pitch.

    A quadratic Bézier curve is used so that overlapping edges remain
    distinguishable.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x_start, y_start, x_end, y_end : float
        Start and end coordinates of the edge.
    weight : float, default 1.0
        Line width representing the strength of the connection.
    color_main : str, optional
        Colour of the main line, defaults to ``COLORS['blue']``.
    color_bg : str, optional
        Background colour drawn beneath the main line, defaults to
        ``COLORS['fog']``.
    curvature : float, default 0.2
        Curvature factor; higher values yield more pronounced curves.
    alpha : float, default 0.6
        Opacity of the main line.
    zorder : int, default 2
        Matplotlib z-order for the patches.
    """
    color_main = color_main or COLORS["blue"]
    color_bg = color_bg or COLORS["fog"]

    cx = (x_start + x_end) / 2.0
    cy = (y_start + y_end) / 2.0
    dx = x_end - x_start
    dy = y_end - y_start
    cx -= dy * curvature
    cy += dx * curvature

    path_data = [
        (Path.MOVETO, (x_start, y_start)),
        (Path.CURVE3, (cx, cy)),
        (Path.CURVE3, (x_end, y_end)),
    ]
    codes, verts = zip(*path_data)
    curve = Path(verts, codes)

    if color_bg:
        bg = PathPatch(curve, linewidth=weight + 2, edgecolor=color_bg,
                       facecolor="none", alpha=alpha * 0.5, zorder=zorder)
        ax.add_patch(bg)

    patch = PathPatch(curve, linewidth=weight, edgecolor=color_main,
                      facecolor="none", alpha=alpha, zorder=zorder + 1)
    ax.add_patch(patch)


def shot_on_target_mask(df):
    """Return a boolean mask for shots on target.

    The function searches common column names that may indicate whether a
    shot was on target.  If none are found it falls back to using the goal
    column, assuming that only goals were on target.
    """
    for col in ("is_on_target", "shot_on_target", "on_target"):
        if col in df.columns:
            return df[col].fillna(0).astype(int) == 1
    if "shot_outcome" in df.columns:
        return df["shot_outcome"].str.lower().isin(["on target", "goal"])
    return df.get("is_goal", 0) == 1


def shot_marker_kwargs(on_target: bool, color: str) -> dict:
    """Style parameters for plotting shots based on shot type.

    Parameters
    ----------
    on_target : bool
        Whether the shot was on target.
    color : str
        Base colour associated with the shooting team.

    Returns
    -------
    dict
        Keyword arguments suitable for ``Axes.scatter``.
    """
    marker = "s" if on_target else "o"
    alpha = 0.95 if on_target else 0.7
    lw = 1.0 if on_target else 0.8
    return {
        "marker": marker,
        "color": color,
        "edgecolors": COLORS["fog"],
        "linewidth": lw,
        "alpha": alpha,
    }


def annotate_score(ax, teams, home_goals: int, away_goals: int,
                   position=(0.5, 1.02), color: str | None = None,
                   fontsize: int = 12) -> None:
    """Annotate the final score above an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    teams : list-like
        Pair of team names ``[home, away]``.
    home_goals, away_goals : int
        Goal totals for each team.
    position : tuple, default ``(0.5, 1.02)``
        Axes fraction coordinates of the annotation.
    color : str, optional
        Text colour, defaults to ``COLORS['fog']``.
    fontsize : int, default 12
        Font size of the annotation text.
    """
    text = f"{teams[0]} {home_goals} - {away_goals} {teams[1]}"
    ax.text(
        position[0],
        position[1],
        text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=color or COLORS["fog"],
        fontsize=fontsize,
    )


def annotate_goals_scatter(ax, shots_df, marker_size: float = 200,
                           color: str | None = None) -> None:
    """Highlight goals on a shot map scatter plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing the shot map.
    shots_df : pandas.DataFrame
        DataFrame with ``x``, ``y`` and ``is_goal`` columns and optionally
        ``minute`` for textual annotation.
    marker_size : float, default 200
        Size of the star marker drawn for goals.
    color : str, optional
        Colour of the star marker. Defaults to ``COLORS['goal']``.
    """
    if shots_df is None or shots_df.empty:
        return
    goals = shots_df[shots_df.get("is_goal", 0) == 1]
    if goals.empty:
        return
    goal_color = color or COLORS["goal"]
    ax.scatter(goals["x"], goals["y"], s=marker_size, marker="*",
               color=goal_color, edgecolors="white", linewidths=1.0,
               zorder=5)
    for _, row in goals.iterrows():
        minute = row.get("minute")
        if minute is not None and not np.isnan(minute):
            ax.text(row["x"], row["y"] - 2, f"{int(minute)}'",
                    ha="center", va="top", fontsize=8,
                    color="white", zorder=6)


def annotate_goals_on_xg(ax, shots_df, team: str,
                         color: str | None = None) -> None:
    """Annotate goal moments on an xG race chart for a given team.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing the xG race plot.
    shots_df : pandas.DataFrame
        DataFrame with columns ``team``, ``minute``, ``xg_val`` and
        ``is_goal``.
    team : str
        Team name to filter shots.
    color : str, optional
        Colour for the annotation markers. Defaults to ``COLORS['goal']``.
    """
    sub = shots_df[shots_df["team"] == team].copy()
    if sub.empty:
        return
    sub = sub.sort_values("minute")
    sub["cum_xg"] = sub["xg_val"].cumsum()
    goals = sub[sub.get("is_goal", 0) == 1]
    if goals.empty:
        return
    ann_color = color or COLORS["goal"]
    ax.scatter(goals["minute"], goals["cum_xg"], s=80, marker="o",
               facecolors="none", edgecolors=ann_color, linewidths=1.5,
               zorder=5)
    for _, row in goals.iterrows():
        ax.text(row["minute"], row["cum_xg"] + 0.02,
                f"{int(row['minute'])}'", color=ann_color,
                ha="center", va="bottom", fontsize=8, zorder=6)
