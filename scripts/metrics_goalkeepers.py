"""Goalkeeping metrics for football event data.

This module provides basic goalkeeping metrics calculated from event streams.
Functions operate on DataFrames containing at minimum the columns
``shot_psxg``, ``shot_outcome``, ``under_pressure``, ``pass_length`` and
``goalkeeper``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def psxg_minus_ga(events: pd.DataFrame) -> pd.DataFrame:
    """Post-shot xG minus goals allowed for each goalkeeper.

    Parameters
    ----------
    events:
        Event stream with ``shot_psxg``, ``shot_outcome`` and ``goalkeeper``
        columns.

    Returns
    -------
    pd.DataFrame
        ``goalkeeper`` and ``psxg_minus_ga`` columns.
    """
    shots = events.dropna(subset=["shot_psxg"]).copy()
    if shots.empty:
        return pd.DataFrame(columns=["goalkeeper", "psxg_minus_ga"])

    if shots["shot_outcome"].dtype.kind in "biufc":
        shots["goals"] = shots["shot_outcome"].astype(float)
    else:
        shots["goals"] = shots["shot_outcome"].str.lower().eq("goal").astype(int)

    agg = shots.groupby("goalkeeper").agg(psxg=("shot_psxg", "sum"), goals=("goals", "sum"))
    agg["psxg_minus_ga"] = agg["psxg"] - agg["goals"]
    return agg[["psxg_minus_ga"]].reset_index()


def shot_stopping_pct(events: pd.DataFrame) -> pd.DataFrame:
    """Shot-stopping percentage based on post-shot xG.

    Calculated as ``(PSxG - goals) / PSxG`` for each goalkeeper.

    Parameters
    ----------
    events:
        Event stream with ``shot_psxg``, ``shot_outcome`` and ``goalkeeper``
        columns.

    Returns
    -------
    pd.DataFrame
        ``goalkeeper`` and ``shot_stopping_pct`` columns.
    """
    shots = events.dropna(subset=["shot_psxg"]).copy()
    if shots.empty:
        return pd.DataFrame(columns=["goalkeeper", "shot_stopping_pct"])

    if shots["shot_outcome"].dtype.kind in "biufc":
        shots["goals"] = shots["shot_outcome"].astype(float)
    else:
        shots["goals"] = shots["shot_outcome"].str.lower().eq("goal").astype(int)

    agg = shots.groupby("goalkeeper").agg(psxg=("shot_psxg", "sum"), goals=("goals", "sum"))
    agg["shot_stopping_pct"] = (agg["psxg"] - agg["goals"]) / agg["psxg"].replace(0, np.nan)
    return agg[["shot_stopping_pct"]].reset_index()


def cross_claims(events: pd.DataFrame) -> pd.DataFrame:
    """Count cross claims for each goalkeeper.

    The function treats rows where ``under_pressure`` equals one as cross
    claim actions.
    """
    keepers = events[["goalkeeper"]].drop_duplicates()
    claims = (
        events[events.get("under_pressure", 0) == 1]
        .groupby("goalkeeper")
        .size()
        .rename("cross_claims")
    )
    out = keepers.merge(claims, on="goalkeeper", how="left").fillna({"cross_claims": 0})
    return out


def sweeper_actions(events: pd.DataFrame, threshold: float = 30.0) -> pd.DataFrame:
    """Count long passes by goalkeepers as a proxy for sweeper actions.

    Parameters
    ----------
    events:
        Event stream with ``pass_length`` and ``goalkeeper`` columns.
    threshold:
        Minimum ``pass_length`` for an action to be considered a sweeper
        action.  Defaults to 30 metres.
    """
    keepers = events[["goalkeeper"]].drop_duplicates()
    cond = events.get("pass_length", 0) >= threshold
    agg = events[cond].groupby("goalkeeper").size().rename("sweeper_actions")
    out = keepers.merge(agg, on="goalkeeper", how="left").fillna({"sweeper_actions": 0})
    return out


def distribution_under_pressure(events: pd.DataFrame) -> pd.DataFrame:
    """Average pass length when under pressure for each goalkeeper."""
    cond = events.get("under_pressure", 0) == 1
    agg = (
        events[cond]
        .groupby("goalkeeper")["pass_length"]
        .mean()
        .rename("distribution_under_pressure")
    )
    keepers = events[["goalkeeper"]].drop_duplicates()
    out = keepers.merge(agg, on="goalkeeper", how="left").fillna(
        {"distribution_under_pressure": 0}
    )
    return out
