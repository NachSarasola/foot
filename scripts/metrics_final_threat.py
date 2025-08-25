"""Final third threat metrics for football event data.

This module implements a small collection of metrics used to quantify
how dangerous a team is in the final third of the pitch.  The functions
expect event data normalised to a pitch of 120×80 yards following the
StatsBomb convention.  Returned values are aggregated by match and team
and are expressed per 90 minutes.
"""
from __future__ import annotations

import pandas as pd

# Pitch and zone definitions -------------------------------------------------
PITCH_LENGTH = 120
PITCH_WIDTH = 80
PENALTY_X = 102  # start of the penalty box on the x-axis
PENALTY_Y_MIN = 18
PENALTY_Y_MAX = 62
ZONE14_X_MIN = 78
ZONE14_X_MAX = 102
ZONE14_Y_MIN = 30
ZONE14_Y_MAX = 50
DEEP_COMPLETION_X = 100  # within 20m (units) of the goal line


def _minutes_played(events: pd.DataFrame) -> pd.Series:
    """Return the minutes played for each match.

    The function assumes that the ``minute`` column is measured from the
    start of the match.  The maximum value found for a match is considered
    the total minutes played for that match.
    """
    return events.groupby("match_id")["minute"].max()


def _scale_per90(df: pd.DataFrame, minutes: pd.Series, column: str) -> pd.DataFrame:
    """Scale metric column to a per 90 minutes rate."""
    out = df.merge(minutes.rename("minutes"), on="match_id", how="left")
    out[column] = out[column] / out["minutes"] * 90
    return out.drop(columns="minutes")


# ---------------------------------------------------------------------------
# xG and xA
# ---------------------------------------------------------------------------

def calculate_xg(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate expected goals (xG) per 90 minutes.

    Shots are identified via the ``is_shot`` column and their expected goals
    value is taken from ``xg``.  The result is aggregated by ``match_id`` and
    ``team`` and scaled to a 90 minute rate.
    """
    teams = events[["match_id", "team"]].drop_duplicates()
    shots = events[events.get("is_shot", 0) == 1]
    agg = shots.groupby(["match_id", "team"])["xg"].sum().reset_index()
    agg = teams.merge(agg, on=["match_id", "team"], how="left").fillna({"xg": 0})
    minutes = _minutes_played(events)
    agg = _scale_per90(agg, minutes, "xg")
    return agg.rename(columns={"xg": "xg_per90"})


def calculate_xa(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate expected assists (xA) per 90 minutes.

    Passes carrying an ``xa`` value are summed for each team and match and
    normalised to a 90 minute rate.
    """
    teams = events[["match_id", "team"]].drop_duplicates()
    passes = events[events.get("xa", 0) > 0]
    agg = passes.groupby(["match_id", "team"])["xa"].sum().reset_index()
    agg = teams.merge(agg, on=["match_id", "team"], how="left").fillna({"xa": 0})
    minutes = _minutes_played(events)
    agg = _scale_per90(agg, minutes, "xa")
    return agg.rename(columns={"xa": "xa_per90"})


# ---------------------------------------------------------------------------
# Shot creating actions
# ---------------------------------------------------------------------------

def shot_creating_actions(events: pd.DataFrame) -> pd.DataFrame:
    """Count shot-creating actions (SCA) per 90 minutes.

    A shot-creating action is defined as the last one or two on-ball events
    (pass or carry) by the shooting team immediately preceding a shot.  This
    is a simplified approximation that relies on event order in the
    DataFrame.
    """
    df = events.sort_values(["match_id", "minute"]).reset_index(drop=True)
    df["prev_team"] = df.groupby("match_id")["team"].shift(1)
    df["prev_is_pass"] = df.groupby("match_id")["is_pass"].shift(1).fillna(0)
    df["prev_is_carry"] = df.groupby("match_id")["is_carry"].shift(1).fillna(0)
    df["prev2_team"] = df.groupby("match_id")["team"].shift(2)
    df["prev2_is_pass"] = df.groupby("match_id")["is_pass"].shift(2).fillna(0)
    df["prev2_is_carry"] = df.groupby("match_id")["is_carry"].shift(2).fillna(0)

    cond1 = (
        (df.get("is_shot", 0) == 1)
        & (df["team"] == df["prev_team"])
        & ((df["prev_is_pass"] == 1) | (df["prev_is_carry"] == 1))
    )
    cond2 = (
        (df.get("is_shot", 0) == 1)
        & (df["team"] == df["prev_team"])
        & ((df["prev_is_pass"] == 1) | (df["prev_is_carry"] == 1))
        & (df["team"] == df["prev2_team"])
        & ((df["prev2_is_pass"] == 1) | (df["prev2_is_carry"] == 1))
    )
    df["sca"] = cond1.astype(int) + cond2.astype(int)

    agg = df.groupby(["match_id", "team"])["sca"].sum().reset_index()
    minutes = _minutes_played(events)
    agg = _scale_per90(agg, minutes, "sca")
    return agg.rename(columns={"sca": "sca_per90"})


# ---------------------------------------------------------------------------
# Completions and entries
# ---------------------------------------------------------------------------

def deep_completions(events: pd.DataFrame) -> pd.DataFrame:
    """Count deep completions per 90 minutes.

    A deep completion is a completed pass that ends within 20 metres of the
    opponent's goal (``end_x >= 100``) but outside the penalty area defined
    by ``x >= 102`` and ``18 <= y <= 62``.
    """
    cond = (
        (events.get("is_pass", 0) == 1)
        & (events["end_x"] >= DEEP_COMPLETION_X)
        & ~(
            (events["end_x"] >= PENALTY_X)
            & (events["end_y"].between(PENALTY_Y_MIN, PENALTY_Y_MAX))
        )
    )
    teams = events[["match_id", "team"]].drop_duplicates()
    agg = events[cond].groupby(["match_id", "team"]).size().reset_index(name="deep")
    agg = teams.merge(agg, on=["match_id", "team"], how="left").fillna({"deep": 0})
    minutes = _minutes_played(events)
    agg = _scale_per90(agg, minutes, "deep")
    return agg.rename(columns={"deep": "deep_completions_per90"})


def box_entries(events: pd.DataFrame) -> pd.DataFrame:
    """Count entries into the penalty area per 90 minutes.

    Any pass or carry that ends inside the opponent's penalty box
    (``x >= 102`` and ``18 <= y <= 62``) is considered a box entry.
    """
    cond = (
        ((events.get("is_pass", 0) == 1) | (events.get("is_carry", 0) == 1))
        & (events["end_x"] >= PENALTY_X)
        & (events["end_y"].between(PENALTY_Y_MIN, PENALTY_Y_MAX))
    )
    teams = events[["match_id", "team"]].drop_duplicates()
    agg = events[cond].groupby(["match_id", "team"]).size().reset_index(name="entries")
    agg = teams.merge(agg, on=["match_id", "team"], how="left").fillna({"entries": 0})
    minutes = _minutes_played(events)
    agg = _scale_per90(agg, minutes, "entries")
    return agg.rename(columns={"entries": "box_entries_per90"})


def zone14_entries(events: pd.DataFrame) -> pd.DataFrame:
    """Count entries into Zone 14 per 90 minutes.

    Zone 14 is defined as the rectangle centrally located just outside the
    penalty area: ``78 <= x <= 102`` and ``30 <= y <= 50``.  Passes or
    carries that end within this zone are counted.
    """
    cond = (
        ((events.get("is_pass", 0) == 1) | (events.get("is_carry", 0) == 1))
        & (events["end_x"].between(ZONE14_X_MIN, ZONE14_X_MAX))
        & (events["end_y"].between(ZONE14_Y_MIN, ZONE14_Y_MAX))
    )
    teams = events[["match_id", "team"]].drop_duplicates()
    agg = events[cond].groupby(["match_id", "team"]).size().reset_index(name="z14")
    agg = teams.merge(agg, on=["match_id", "team"], how="left").fillna({"z14": 0})
    minutes = _minutes_played(events)
    agg = _scale_per90(agg, minutes, "z14")
    return agg.rename(columns={"z14": "zone14_entries_per90"})


def passes_to_penalty_area(events: pd.DataFrame) -> pd.DataFrame:
    """Count passes that end inside the penalty area per 90 minutes."""
    cond = (
        (events.get("is_pass", 0) == 1)
        & (events["end_x"] >= PENALTY_X)
        & (events["end_y"].between(PENALTY_Y_MIN, PENALTY_Y_MAX))
    )
    teams = events[["match_id", "team"]].drop_duplicates()
    agg = events[cond].groupby(["match_id", "team"]).size().reset_index(name="ppa")
    agg = teams.merge(agg, on=["match_id", "team"], how="left").fillna({"ppa": 0})
    minutes = _minutes_played(events)
    agg = _scale_per90(agg, minutes, "ppa")
    return agg.rename(columns={"ppa": "passes_to_penalty_area_per90"})


# ---------------------------------------------------------------------------
# Set-piece threat
# ---------------------------------------------------------------------------

def set_piece_xg(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate xG from set pieces per 90 minutes.

    Shots flagged with ``is_set_piece`` are considered set-piece shots.
    Their xG values are summed and scaled to a 90 minute rate.
    """
    shots = events[(events.get("is_shot", 0) == 1) & (events.get("is_set_piece", 0) == 1)]
    teams = events[["match_id", "team"]].drop_duplicates()
    agg = shots.groupby(["match_id", "team"])["xg"].sum().reset_index()
    agg = teams.merge(agg, on=["match_id", "team"], how="left").fillna({"xg": 0})
    minutes = _minutes_played(events)
    agg = _scale_per90(agg, minutes, "xg")
    return agg.rename(columns={"xg": "set_piece_xg_per90"})
