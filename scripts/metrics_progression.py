"""Ball progression and possession value metrics.

This module provides simplified implementations of a number of popular
progression metrics.  The intent is to supply light‑weight helpers that work
with the synthetic event data used in the tests.  The pitch is assumed to be
120×80 and coordinates follow the StatsBomb convention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PITCH_LENGTH = 120
PITCH_WIDTH = 80


def _minutes_played(events: pd.DataFrame) -> pd.Series:
    """Return the minutes played for each match."""
    return events.groupby("match_id")["minute"].max()


def _scale_per90(df: pd.DataFrame, minutes: pd.Series, column: str) -> pd.DataFrame:
    """Scale metric column to a per 90 minutes rate."""
    out = df.merge(minutes.rename("minutes"), on="match_id", how="left")
    out[column] = out[column] / out["minutes"] * 90
    return out.drop(columns="minutes")


# ---------------------------------------------------------------------------
# Loading and looking up the open‑xT grid
# ---------------------------------------------------------------------------


def load_xt_grid(path: str | Path) -> np.ndarray:
    """Load a 12×8 open‑xT grid from ``path``.

    The grid is expected to be stored as a CSV file with 8 rows (y
    direction) and 12 columns (x direction).  Values should correspond to the
    open‑xT v1 model popularised by Karun Singh.  The function returns the
    grid as a :class:`numpy.ndarray` for fast lookups.
    """
    grid = pd.read_csv(path, header=None)
    if grid.shape != (8, 12):  # pragma: no cover - defensive programming
        raise ValueError("xT grid must be 8×12 cells")
    return grid.values


def lookup_xt(x: float, y: float, grid: np.ndarray) -> float:
    """Return the xT value at coordinates ``(x, y)``.

    Coordinates are mapped to the grid by dividing the pitch into 12×8 equally
    sized cells.  Values falling outside the pitch are clipped to the nearest
    cell.
    """
    x_bin = int(np.clip(x, 0, PITCH_LENGTH - 1) // (PITCH_LENGTH / 12))
    y_bin = int(np.clip(y, 0, PITCH_WIDTH - 1) // (PITCH_WIDTH / 8))
    return float(grid[y_bin, x_bin])


def xt_lookup(events: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Add start and end xT values for passes and carries."""
    df = events.copy()
    mask = (df.get("is_pass", 0) == 1) | (df.get("is_carry", 0) == 1)
    df.loc[mask, "start_xt"] = [lookup_xt(x, y, grid) for x, y in df.loc[mask, ["x", "y"]].to_numpy()]
    df.loc[mask, "end_xt"] = [lookup_xt(x, y, grid) for x, y in df.loc[mask, ["end_x", "end_y"]].to_numpy()]
    df.loc[~mask, "start_xt"] = 0.0
    df.loc[~mask, "end_xt"] = 0.0
    return df


def xt_added(events: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Calculate xT added per 90 minutes for each team and match."""
    df = xt_lookup(events, grid)
    df["xt_added"] = df.get("end_xt", 0) - df.get("start_xt", 0)
    mask = (df.get("is_pass", 0) == 1) | (df.get("is_carry", 0) == 1)
    agg = (
        df[mask]
        .groupby(["match_id", "team"])["xt_added"]
        .sum()
        .reset_index()
    )
    teams = events[["match_id", "team"]].drop_duplicates()
    agg = teams.merge(agg, on=["match_id", "team"], how="left").fillna({"xt_added": 0})
    minutes = _minutes_played(events)
    agg = _scale_per90(agg, minutes, "xt_added")
    return agg


# ---------------------------------------------------------------------------
# Progressive actions
# ---------------------------------------------------------------------------

PROGRESS_PASS_THRESHOLD = 15
PROGRESS_CARRY_THRESHOLD = 10


def progressive_pass(events: pd.DataFrame) -> pd.DataFrame:
    """Count progressive passes per 90 minutes."""
    cond = (
        (events.get("is_pass", 0) == 1)
        & ((events["end_x"] - events["x"]) >= PROGRESS_PASS_THRESHOLD)
        & (events["end_x"] > events["x"])
    )
    teams = events[["match_id", "team"]].drop_duplicates()
    agg = events[cond].groupby(["match_id", "team"]).size().reset_index(name="prog_pass")
    agg = teams.merge(agg, on=["match_id", "team"], how="left").fillna({"prog_pass": 0})
    minutes = _minutes_played(events)
    agg = _scale_per90(agg, minutes, "prog_pass")
    return agg.rename(columns={"prog_pass": "progressive_passes_per90"})


def progressive_carry(events: pd.DataFrame) -> pd.DataFrame:
    """Count progressive carries per 90 minutes."""
    cond = (
        (events.get("is_carry", 0) == 1)
        & ((events["end_x"] - events["x"]) >= PROGRESS_CARRY_THRESHOLD)
        & (events["end_x"] > events["x"])
    )
    teams = events[["match_id", "team"]].drop_duplicates()
    agg = events[cond].groupby(["match_id", "team"]).size().reset_index(name="prog_carry")
    agg = teams.merge(agg, on=["match_id", "team"], how="left").fillna({"prog_carry": 0})
    minutes = _minutes_played(events)
    agg = _scale_per90(agg, minutes, "prog_carry")
    return agg.rename(columns={"prog_carry": "progressive_carries_per90"})


# ---------------------------------------------------------------------------
# Possession value metrics
# ---------------------------------------------------------------------------

def epv_simplified(events: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Estimate expected possession value (EPV) per 90 minutes.

    Passes and carries are valued by the xT they add while shots are valued by
    their xG.  The result is aggregated by team and match and scaled to a per
    90 minutes rate.
    """
    df = xt_lookup(events, grid)
    df["value"] = 0.0
    mask_action = (df.get("is_pass", 0) == 1) | (df.get("is_carry", 0) == 1)
    df.loc[mask_action, "value"] = df.loc[mask_action, "end_xt"] - df.loc[mask_action, "start_xt"]
    df.loc[df.get("is_shot", 0) == 1, "value"] = df.loc[df.get("is_shot", 0) == 1, "xg"].astype(float)
    agg = df.groupby(["match_id", "team"])["value"].sum().reset_index()
    minutes = _minutes_played(events)
    agg = _scale_per90(agg, minutes, "value")
    return agg.rename(columns={"value": "epv_per90"})


def _assign_possessions(events: pd.DataFrame) -> pd.DataFrame:
    """Return events with a simple possession id assigned."""
    df = events.sort_values(["match_id", "minute"]).reset_index(drop=True)
    df["possession"] = (df["team"] != df["team"].shift(1)).cumsum()
    return df


def xchain(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate xGChain per 90 minutes."""
    df = _assign_possessions(events)
    df["shot_xg"] = df.groupby("possession")["xg"].transform("sum")
    agg = df.groupby(["match_id", "team"])["shot_xg"].sum().reset_index()
    minutes = _minutes_played(events)
    agg = _scale_per90(agg, minutes, "shot_xg")
    return agg.rename(columns={"shot_xg": "xg_chain_per90"})


def xbuildup(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate xGBuildup per 90 minutes.

    Shots and the action immediately preceding a shot are excluded from the
    buildup total.
    """
    df = _assign_possessions(events)
    df["shot_xg"] = df.groupby("possession")["xg"].transform("sum")
    # mark key action immediately before shot
    df["is_key_action"] = False
    shot_indices = df.index[df.get("is_shot", 0) == 1]
    df.loc[shot_indices - 1, "is_key_action"] = df.loc[shot_indices - 1, "team"] == df.loc[shot_indices, "team"].values
    cond = (~df.get("is_shot", 0).astype(bool)) & (~df["is_key_action"])
    agg = df[cond].groupby(["match_id", "team"])["shot_xg"].sum().reset_index()
    minutes = _minutes_played(events)
    agg = _scale_per90(agg, minutes, "shot_xg")
    return agg.rename(columns={"shot_xg": "xg_buildup_per90"})
