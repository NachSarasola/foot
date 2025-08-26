"""Utility functions and styling helpers for Ush Analytics plots.

This module centralises the colour palette and a set of helpers that are
shared between scripts.  The functions are lightweight wrappers built on
matplotlib and mplsoccer so they can be reused in notebooks or other
visualisation pipelines.
"""

from __future__ import annotations

from pathlib import Path as FilePath

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "navy": "#0A2540",
    "cyan": "#00C2FF",
    "blue": "#5B86E5",
    "fog": "#E6EEF6",
    "paper": "#FFFFFF",
    "grass": "#0B1E2D",
    "goal": "#FF3366",
    "ink": "#0A2540",
}

# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

__all__ = [
    "COLORS",
    "add_grass_texture",
    "scale_sizes",
    "label",
    "text_halo",
    "avoid_overlap",
    "edge_curved",
    "shot_on_target_mask",
    "shot_marker_kwargs",
    "annotate_score",
    "annotate_goals_scatter",
    "annotate_goals_on_xg",
    "save_fig_pro",
]


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
          **kwargs):
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
    return ax.text(x, y - y_offset, text, ha="center", va="top",
                   fontsize=fontsize, color=color or COLORS["navy"],
                   zorder=zorder, **kwargs)

def text_halo(ax, text: str, **kwargs):
    """Draw text with a semi-transparent white background box.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    text : str
        Text to display.
    **kwargs :
        Must include ``x`` and ``y`` coordinates along with any other
        arguments accepted by :meth:`matplotlib.axes.Axes.text`.
    """
    try:
        x = kwargs.pop("x")
        y = kwargs.pop("y")
    except KeyError as exc:  # pragma: no cover - defensive
        raise TypeError("text_halo requires 'x' and 'y' keyword arguments") from exc

    bbox = kwargs.pop("bbox", None) or {
        "boxstyle": "round,pad=0.2",
        "facecolor": COLORS["paper"],
        "alpha": 0.7,
        "edgecolor": "none",
    }
    return ax.text(x, y, text, bbox=bbox, **kwargs)


def avoid_overlap(labels, padding: float = 5) -> None:
    """Nudge text labels vertically to minimise overlaps."""
    if not labels:
        return
    ax = labels[0].axes
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    moved = True
    while moved:
        moved = False
        bboxes = [t.get_window_extent(renderer=renderer) for t in labels]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if bboxes[i].overlaps(bboxes[j]):
                    x, y = labels[j].get_position()
                    x_disp, y_disp = ax.transData.transform((x, y))
                    y_disp += padding
                    x_new, y_new = ax.transData.inverted().transform((x_disp, y_disp))
                    labels[j].set_position((x_new, y_new))
                    renderer = fig.canvas.get_renderer()
                    bboxes[j] = labels[j].get_window_extent(renderer=renderer)
                    moved = True


def edge_curved(ax, start: tuple[float, float], end: tuple[float, float],
                weight: float = 1.0, color: str | None = None,
                shadow: bool = False, shadow_color: str | None = None,
                curvature: float = 0.2, alpha: float = 0.6,
                zorder: int = 2) -> None:
    """Draw a curved edge between two nodes with optional shadow."""
    color = color or COLORS["blue"]
    shadow_color = shadow_color or COLORS["fog"]

    x_start, y_start = start
    x_end, y_end = end
    cx = (x_start + x_end) / 2.0
    cy = (y_start + y_end) / 2.0
    dx = x_end - x_start
    dy = y_end - y_start
    cx -= dy * curvature
    cy += dx * curvature

    path_data = [
        (MplPath.MOVETO, (x_start, y_start)),
        (MplPath.CURVE3, (cx, cy)),
        (MplPath.CURVE3, (x_end, y_end)),
    ]
    codes, verts = zip(*path_data)
    curve = MplPath(verts, codes)

    if shadow:
        bg = PathPatch(
            curve,
            linewidth=weight + 2,
            edgecolor=shadow_color,
            facecolor="none",
            alpha=alpha * 0.5,
            capstyle="round",
            zorder=zorder,
        )
        ax.add_patch(bg)

    patch = PathPatch(
        curve,
        linewidth=weight,
        edgecolor=color,
        facecolor="none",
        alpha=alpha,
        capstyle="round",
        zorder=zorder + (1 if shadow else 0),
    )
    ax.add_patch(patch)


curved_edge = edge_curved


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


def save_fig_pro(fig, path, px: tuple[int, int] = (1600, 1000),
                 dpi: int = 240) -> None:
    """Save figure in standard Ush resolution plus thumbnail.

    The figure is saved to ``path`` and a reduced thumbnail named
    ``<stem>_thumb.png`` is generated alongside it.
    """
    outfile = FilePath(path)
    width, height = px
    pad = 20
    orig_size = fig.get_size_inches()
    fig.set_size_inches((width - 2 * pad) / dpi, (height - 2 * pad) / dpi)
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight",
                pad_inches=pad / dpi, facecolor=COLORS["navy"])

    thumb = outfile.with_name(f"{outfile.stem}_thumb.png")
    fig.set_size_inches((800 - 2 * pad) / dpi, (500 - 2 * pad) / dpi)
    fig.savefig(thumb, dpi=dpi, bbox_inches="tight",
                pad_inches=pad / dpi, facecolor=COLORS["navy"])

    fig.set_size_inches(orig_size)
