"""Pressing metrics for football event data.

This module provides basic pressing related metrics calculated from event
streams.  Functions operate on DataFrames containing at minimum the columns
``team``, ``period``, ``is_pass``, ``is_def_action``, ``turnover``,
``recovery``, ``tackle``, ``block``, ``y`` and ``possession_id``.  Results are
aggregated by team and period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HIGH_Y = 60  # start of the high pressing zone


def _teams_periods(events: pd.DataFrame) -> pd.DataFrame:
    """Return unique team/period combinations present in ``events``."""
    return events[["team", "period"]].drop_duplicates()


def ppda(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate passes allowed per defensive action (PPDA).

    For each team and period the number of opposition passes is divided by the
    number of defensive actions performed by the team.
    """
    teams = _teams_periods(events)
    passes = events[events.get("is_pass", 0) == 1]
    passes_team = passes.groupby(["team", "period"]).size().rename("team_passes")
    passes_total = passes.groupby("period").size().rename("total_passes")
    def_actions = (
        events[events.get("is_def_action", 0) == 1]
        .groupby(["team", "period"])  # type: ignore[arg-type]
        .size()
        .rename("def_actions")
    )
    out = (
        teams.merge(passes_total, on="period", how="left")
        .merge(passes_team, on=["team", "period"], how="left")
        .merge(def_actions, on=["team", "period"], how="left")
        .fillna({"team_passes": 0, "def_actions": 0})
    )
    out["opp_passes"] = out["total_passes"] - out["team_passes"]
    out["ppda"] = out["opp_passes"] / out["def_actions"].replace(0, np.nan)
    return out[["team", "period", "ppda"]]


def oppda(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate opponent PPDA (OPPDA).

    This is simply the PPDA from the opponent's perspective – passes completed
    by the team divided by the defensive actions of their opponents.
    """
    teams = _teams_periods(events)
    passes = events[events.get("is_pass", 0) == 1]
    passes_team = passes.groupby(["team", "period"]).size().rename("team_passes")
    def_actions = (
        events[events.get("is_def_action", 0) == 1]
        .groupby(["team", "period"])  # type: ignore[arg-type]
        .size()
        .rename("team_def_actions")
    )
    def_total = def_actions.groupby("period").sum().rename("total_def")
    out = (
        teams.merge(passes_team, on=["team", "period"], how="left")
        .merge(def_total, on="period", how="left")
        .merge(def_actions, on=["team", "period"], how="left")
        .fillna({"team_passes": 0, "team_def_actions": 0})
    )
    out["opp_def_actions"] = out["total_def"] - out["team_def_actions"]
    out["oppda"] = out["team_passes"] / out["opp_def_actions"].replace(0, np.nan)
    return out[["team", "period", "oppda"]]


def high_regains(events: pd.DataFrame) -> pd.DataFrame:
    """Count regains in the high press zone."""
    is_regain = (
        (events.get("recovery", 0) == 1)
        | (events.get("tackle", 0) == 1)
        | (events.get("block", 0) == 1)
    )
    cond = is_regain & (events.get("y", 0) >= HIGH_Y)
    agg = events[cond].groupby(["team", "period"]).size().reset_index(name="high_regains")
    teams = _teams_periods(events)
    out = teams.merge(agg, on=["team", "period"], how="left").fillna({"high_regains": 0})
    return out


def counterpress_regains(events: pd.DataFrame) -> pd.DataFrame:
    """Count high regains within three actions after a team's own turnover."""
    df = events.reset_index(drop=True).copy()
    df["is_regain"] = (
        (df.get("recovery", 0) == 1)
        | (df.get("tackle", 0) == 1)
        | (df.get("block", 0) == 1)
    )
    df["last_turnover_team"] = df["team"].where(df.get("turnover", 0) == 1).ffill()
    df["last_turnover_idx"] = pd.Series(df.index, index=df.index).where(df.get("turnover", 0) == 1).ffill()
    df["since_turnover"] = df.index - df["last_turnover_idx"].astype(float)
    cond = (
        df["is_regain"]
        & (df.get("y", 0) >= HIGH_Y)
        & (df["team"] == df["last_turnover_team"])
        & (df["since_turnover"] <= 3)
    )
    agg = df[cond].groupby(["team", "period"]).size().reset_index(name="counterpress_regains")
    teams = _teams_periods(df)
    out = teams.merge(agg, on=["team", "period"], how="left").fillna({"counterpress_regains": 0})
    return out


def high_turnovers(events: pd.DataFrame) -> pd.DataFrame:
    """Count opponent turnovers won in the high zone."""
    df = events.reset_index(drop=True).copy()
    df["next_team"] = df["team"].shift(-1)
    df["next_period"] = df["period"].shift(-1)
    cond = (
        (df.get("turnover", 0) == 1)
        & (df.get("y", 0) >= HIGH_Y)
        & (df["next_team"] != df["team"])
        & (df["period"] == df["next_period"])
    )
    agg = df[cond].groupby(["next_team", "period"]).size().reset_index(name="high_turnovers")
    teams = _teams_periods(events)
    out = (
        teams.merge(agg, left_on=["team", "period"], right_on=["next_team", "period"], how="left")
        .drop(columns="next_team")
        .fillna({"high_turnovers": 0})
    )
    return out


def press_intensity(events: pd.DataFrame) -> pd.DataFrame:
    """Defensive actions in the high zone per opposition pass."""
    teams = _teams_periods(events)
    def_high = (
        events[
            (events.get("is_def_action", 0) == 1) & (events.get("y", 0) >= HIGH_Y)
        ]
        .groupby(["team", "period"])  # type: ignore[arg-type]
        .size()
        .rename("def_high")
    )
    passes = events[events.get("is_pass", 0) == 1]
    passes_team = passes.groupby(["team", "period"]).size().rename("team_passes")
    passes_total = passes.groupby("period").size().rename("total_passes")
    out = (
        teams.merge(def_high, on=["team", "period"], how="left")
        .merge(passes_team, on=["team", "period"], how="left")
        .merge(passes_total, on="period", how="left")
        .fillna({"def_high": 0, "team_passes": 0})
    )
    out["opp_passes"] = out["total_passes"] - out["team_passes"]
    out["press_intensity"] = out["def_high"] / out["opp_passes"].replace(0, np.nan)
    return out[["team", "period", "press_intensity"]]


def field_tilt(events: pd.DataFrame) -> pd.DataFrame:
    """Share of final-third passes completed by each team."""
    teams = _teams_periods(events)
    final_third_passes = (
        events[(events.get("is_pass", 0) == 1) & (events.get("y", 0) >= HIGH_Y)]
        .groupby(["team", "period"])  # type: ignore[arg-type]
        .size()
        .rename("att_passes")
    )
    total_final = (
        events[(events.get("is_pass", 0) == 1) & (events.get("y", 0) >= HIGH_Y)]
        .groupby("period")
        .size()
        .rename("total_att")
    )
    out = (
        teams.merge(final_third_passes, on=["team", "period"], how="left")
        .merge(total_final, on="period", how="left")
        .fillna({"att_passes": 0})
    )
    out["field_tilt"] = out["att_passes"] / out["total_att"].replace(0, np.nan)
    return out[["team", "period", "field_tilt"]]
