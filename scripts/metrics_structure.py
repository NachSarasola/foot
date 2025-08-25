"""Structural metrics for event data.

This module implements simplified versions of a number of classic tactical
structure metrics used in football analysis.  The functions are intentionally
light‑weight and operate on pandas DataFrames with a minimal set of columns so
that they work with the synthetic data employed in the unit tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Pitch dimensions follow the StatsBomb convention used elsewhere in the
# repository.  They are included here to avoid importing the progression
# module which defines the same constants.
PITCH_LENGTH = 120
PITCH_WIDTH = 80


def pass_network(
    events: pd.DataFrame, team: str, grid: np.ndarray | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return passing network nodes and edges for ``team``.

    Parameters
    ----------
    events:
        Event stream containing at least ``team``, ``player``, ``receiver``,
        ``x``, ``y``, ``end_x`` and ``end_y`` as columns.  Only rows with
        ``is_pass`` equal to one are used.
    team:
        Name of the team for which the network should be constructed.
    grid:
        Optional 12×8 open‑xT grid.  When supplied the xT added for each edge
        is computed using a simple lookup similar to :func:`metrics_progression.lookup_xt`.

    Returns
    -------
    nodes, edges:
        ``nodes`` contains one row per player with their average on‑ball
        coordinates. ``edges`` contains one row per passer/receiver pair with
        pass counts (``weight``) and summed xT added (``xt_added``).
    """
    passes = events[(events.get("is_pass", 0) == 1) & (events["team"] == team)].copy()
    if passes.empty:
        return pd.DataFrame(columns=["player", "x", "y"]), pd.DataFrame(
            columns=["source", "target", "weight", "xt_added"]
        )

    # ------------------------------------------------------------------
    # Nodes – average of starting and ending locations for each player
    # ------------------------------------------------------------------
    start = passes[["player", "x", "y"]].rename(columns={"player": "player", "x": "x", "y": "y"})
    end = passes[["receiver", "end_x", "end_y"]].rename(
        columns={"receiver": "player", "end_x": "x", "end_y": "y"}
    )
    nodes = pd.concat([start, end])
    nodes = nodes.groupby("player", as_index=False)[["x", "y"]].mean()

    # ------------------------------------------------------------------
    # Edges – count passes between players and sum their xT contribution
    # ------------------------------------------------------------------
    edges = passes.groupby(["player", "receiver"]).size().reset_index(name="weight")
    edges = edges.rename(columns={"player": "source", "receiver": "target"})

    if grid is not None:
        # Simple xT lookup using the same method as metrics_progression.lookup_xt
        def _lookup(x: float, y: float) -> float:
            x_bin = int(np.clip(x, 0, PITCH_LENGTH - 1) // (PITCH_LENGTH / 12))
            y_bin = int(np.clip(y, 0, PITCH_WIDTH - 1) // (PITCH_WIDTH / 8))
            return float(grid[y_bin, x_bin])

        passes["xt_added"] = [
            _lookup(ex, ey) - _lookup(sx, sy)
            for sx, sy, ex, ey in passes[["x", "y", "end_x", "end_y"]].to_numpy()
        ]
        xt_edges = (
            passes.groupby(["player", "receiver"])["xt_added"].sum().reset_index()
        )
        xt_edges = xt_edges.rename(columns={"player": "source", "receiver": "target"})
        edges = edges.merge(xt_edges, on=["source", "target"], how="left")
    else:
        edges["xt_added"] = 0.0

    return nodes, edges


def lane_carries(events: pd.DataFrame) -> pd.DataFrame:
    """Count carries occurring in each vertical lane of the pitch.

    The pitch is divided into three equal width lanes: left, centre and right
    (from the attacking team's perspective).  The function returns the number
    of carries starting in each lane for every team and match.
    """
    carries = events[events.get("is_carry", 0) == 1].copy()
    if carries.empty:
        return pd.DataFrame(
            columns=[
                "match_id",
                "team",
                "left_carries",
                "centre_carries",
                "right_carries",
            ]
        )

    def _lane(y: float) -> str:
        if y < PITCH_WIDTH / 3:
            return "left"
        if y < 2 * PITCH_WIDTH / 3:
            return "centre"
        return "right"

    carries["lane"] = carries["y"].apply(_lane)
    agg = carries.groupby(["match_id", "team", "lane"]).size().unstack(fill_value=0)
    agg = agg.rename(columns={"left": "left_carries", "centre": "centre_carries", "right": "right_carries"})
    return agg.reset_index()


def switches_throughballs_cutbacks(events: pd.DataFrame) -> pd.DataFrame:
    """Count selected special pass types per team and match."""
    passes = events[events.get("is_pass", 0) == 1].copy()
    if passes.empty:
        return pd.DataFrame(
            columns=["match_id", "team", "switches", "throughballs", "cutbacks"]
        )

    dy = passes["end_y"] - passes["y"]
    dx = passes["end_x"] - passes["x"]

    passes["is_switch"] = (dy.abs() > (PITCH_WIDTH / 2)) & (dx > 0)
    passes["is_throughball"] = (dx >= 20) & (dy.abs() <= 10)
    passes["is_cutback"] = (
        (passes["x"] >= PITCH_LENGTH * (2 / 3))
        & (dx < 0)
        & ((passes["y"] <= PITCH_WIDTH * 0.2) | (passes["y"] >= PITCH_WIDTH * 0.8))
    )

    agg = (
        passes.groupby(["match_id", "team"])[
            ["is_switch", "is_throughball", "is_cutback"]
        ]
        .sum()
        .reset_index()
    )
    return agg.rename(
        columns={
            "is_switch": "switches",
            "is_throughball": "throughballs",
            "is_cutback": "cutbacks",
        }
    )


defensive_action_mask = "is_def_action"


def defensive_compactness(events: pd.DataFrame) -> pd.DataFrame:
    """Measure defensive compactness as vertical spread of actions."""
    df = events[events.get(defensive_action_mask, 0) == 1]
    if df.empty:
        return pd.DataFrame(columns=["match_id", "team", "defensive_compactness"])
    agg = df.groupby(["match_id", "team"])["y"].agg(lambda s: s.max() - s.min())
    return agg.reset_index(name="defensive_compactness")


def line_height(events: pd.DataFrame) -> pd.DataFrame:
    """Average x‑coordinate of defensive actions for each team and match."""
    df = events[events.get(defensive_action_mask, 0) == 1]
    if df.empty:
        return pd.DataFrame(columns=["match_id", "team", "line_height"])
    agg = df.groupby(["match_id", "team"])["x"].mean().reset_index(name="line_height")
    return agg


if __name__ == "__main__":  # pragma: no cover - manual visual generation
    # Generate a very small demonstration pass network from the bundled
    # events.csv.  This block is not executed during testing but allows the
    # repository to ship with a preview image.
    import matplotlib.pyplot as plt
    from metrics_progression import load_xt_grid

    data = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "events.csv")
    grid = load_xt_grid(Path(__file__).resolve().parents[1] / "data" / "xt_grid.csv")
    team = data["team"].unique()[0]
    nodes, edges = pass_network(data, team=team, grid=grid)

    fig, ax = plt.subplots(figsize=(6.667, 4.167), dpi=240)
    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(0, PITCH_WIDTH)
    ax.set_axis_off()

    for _, row in edges.iterrows():
        src = nodes[nodes.player == row.source].iloc[0]
        tgt = nodes[nodes.player == row.target].iloc[0]
        ax.plot([src.x, tgt.x], [src.y, tgt.y], color="lightgrey", linewidth=row.weight)

    ax.scatter(nodes.x, nodes.y, s=100, color="steelblue", zorder=5)
    for _, row in nodes.iterrows():
        ax.text(row.x, row.y, row.player, ha="center", va="center", color="white")

    out_path = Path(__file__).resolve().parents[1] / "report" / "img" / "pass_network.png"
    fig.savefig(out_path, bbox_inches="tight")
