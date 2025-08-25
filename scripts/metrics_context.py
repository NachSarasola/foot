"""Contextual metrics for football event data.

This module provides helpers to label events with the game state and phase of
play and to aggregate existing per–event metrics (such as xG) by these
contexts.  The goal is to make it easy to analyse performance depending on
whether a team was leading or trailing and in which phase of play the actions
occurred.
"""
from __future__ import annotations

from typing import Iterable

from pathlib import Path

import numpy as np
import pandas as pd

from metrics_pressing import ppda


def validate_events(df: pd.DataFrame) -> None:
    """Validate event data consistency.

    The function raises a :class:`ValueError` if any of the following
    conditions are not met:

    * The number of goals recorded in ``df`` matches the totals from
      ``data/matches.csv``.
    * The :func:`metrics_pressing.ppda` calculation yields only finite
      values.
    * The ``xt_added`` column contains no negative values.
    """

    matches_path = Path(__file__).resolve().parents[1] / "data" / "matches.csv"
    matches = pd.read_csv(matches_path)

    goals_events = int(df["is_goal"].sum())
    goals_matches = int(matches["home_score"].sum() + matches["away_score"].sum())
    if goals_events != goals_matches:
        raise ValueError("Mismatch between event goals and match scores")

    ppda_df = ppda(df)
    if not np.isfinite(ppda_df["ppda"]).all():
        raise ValueError("PPDA contains inf or NaN values")

    if (df["xt_added"] < 0).any():
        raise ValueError("xt_added must be non-negative")


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

    The phase is derived from spatial information and event type using the
    following heuristics (``x`` and ``y`` scaled 0–100 along pitch length and
    width):

    ``build_up``
        Actions occurring in the defensive third (``x < 40``).
    ``progression``
        Actions in the middle third (``40 ≤ x < 80``).
    ``finalization``
        Shots or any action inside the attacking third (``x ≥ 80``).

    A lower‑case ``event_type`` column is consulted so that shot events are
    always labelled as ``finalization`` regardless of their location.  If the
    required columns are missing, zeros or empty strings are assumed.
    """

    df = events.copy()

    x = df.get("x", 0)
    y = df.get("y", 0)
    event_type = df.get("event_type", "").astype(str).str.lower()

    finalization = (x >= 80) | (event_type == "shot") | ((x >= 70) & y.between(30, 70))
    progression = (~finalization) & (x >= 40)

    df["phase"] = np.select(
        [finalization, progression],
        ["finalization", "progression"],
        default="build_up",
    )

    return df


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
    """Aggregate metrics by team and phase of play.

    Phases are the values produced by :func:`phase_of_play` – ``build_up``,
    ``progression`` and ``finalization``.
    """
    df = phase_of_play(events)
    return _aggregate(df, ["phase"], metrics)


def aggregate_context(events: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """Aggregate metrics by period, score state and phase simultaneously.

    The resulting ``phase`` column follows the same build‑up → progression →
    finalization labels as :func:`phase_of_play`.
    """
    df = score_state(phase_of_play(events))
    return _aggregate(df, ["period", "score_state", "phase"], metrics)
