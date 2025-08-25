"""Contextual metrics for football event data.

This module provides helpers to label events with the game state and phase of
play and to aggregate existing per–event metrics (such as xG) by these
contexts.  The goal is to make it easy to analyse performance depending on
whether a team was leading or trailing and in which phase of play the actions
occurred.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def score_state(events: pd.DataFrame) -> pd.DataFrame:
    """Return ``events`` with a ``score_state`` column added.

    The state reflects the score *before* each event from the perspective of
    the acting team.  Values are ``"ahead"``, ``"level"`` or ``"behind"``.  The
    input DataFrame is expected to contain at least the columns ``team`` and
    ``is_goal`` (1 for goals, 0 otherwise).  An optional ``match_id`` column is
    respected so multiple matches can be processed together.
    """
    df = events.copy().reset_index(drop=True)
    df["is_goal"] = df.get("is_goal", 0).astype(int)

    match_cols: list[str] = ["match_id"] if "match_id" in df.columns else []
    # cumulative goals before each event
    group_team = match_cols + ["team"]
    df["team_goals"] = df.groupby(group_team)["is_goal"].cumsum() - df["is_goal"]
    if match_cols:
        df["total_goals"] = df.groupby(match_cols)["is_goal"].cumsum() - df["is_goal"]
    else:
        df["total_goals"] = df["is_goal"].cumsum() - df["is_goal"]
    df["opp_goals"] = df["total_goals"] - df["team_goals"]

    diff = df["team_goals"] - df["opp_goals"]
    df["score_state"] = pd.cut(
        diff,
        bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=["behind", "level", "ahead"],
    )
    return df.drop(columns=["team_goals", "total_goals", "opp_goals"])


def phase_of_play(events: pd.DataFrame) -> pd.DataFrame:
    """Return ``events`` with a ``phase`` column added.

    The phase is a coarse classification using two optional boolean columns:

    ``in_possession``
        1 if the acting team is in possession of the ball, else 0.
    ``is_set_piece``
        1 if the event is a set piece (free‑kick, corner, etc.).

    Phases are labelled ``"attack"`` when ``in_possession`` is true,
    ``"defence"`` when it is false and ``"set_piece"`` takes precedence when
    ``is_set_piece`` is true.
    """
    df = events.copy()
    df["in_possession"] = df.get("in_possession", 0).astype(int)
    df["is_set_piece"] = df.get("is_set_piece", 0).astype(int)

    df["phase"] = np.where(df["in_possession"] == 1, "attack", "defence")
    df.loc[df["is_set_piece"] == 1, "phase"] = "set_piece"
    return df.drop(columns=["in_possession", "is_set_piece"])


def _aggregate(df: pd.DataFrame, group_cols: Iterable[str], metrics: Iterable[str]) -> pd.DataFrame:
    """Aggregate ``metrics`` by ``group_cols`` and ``team``."""
    cols = ["team", *group_cols]
    agg = df.groupby(cols)[list(metrics)].sum().reset_index()
    return agg


def aggregate_by_period(events: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """Aggregate metrics by team and period."""
    return _aggregate(events, ["period"], metrics)


def aggregate_by_score_state(events: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """Aggregate metrics by team and score state."""
    df = score_state(events)
    return _aggregate(df, ["score_state"], metrics)


def aggregate_by_phase(events: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """Aggregate metrics by team and phase of play."""
    df = phase_of_play(events)
    return _aggregate(df, ["phase"], metrics)


def aggregate_context(events: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """Aggregate metrics by period, score state and phase simultaneously."""
    df = score_state(phase_of_play(events))
    return _aggregate(df, ["period", "score_state", "phase"], metrics)
