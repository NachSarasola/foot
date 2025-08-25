"""Normalise pitch coordinates from different providers to [0, 1].

The functions in this module convert provider specific coordinate systems to a
common normalised scale where ``x_norm`` runs from 0 (own goal line) to 1
(opponent goal line) and ``y_norm`` runs from 0 (top touchline) to 1 (bottom
touchline).  By default the attacking team is assumed to play from left to
right.  When the team attacks from right to left the ``x`` coordinate is
reflected so that ``x_norm`` still increases towards the opponent goal.
"""

from __future__ import annotations

import pandas as pd


def to_ush_coords_statsbomb(
    statsbomb_df: pd.DataFrame, attack_direction: str = "ltr"
) -> pd.DataFrame:
    """Return StatsBomb data with normalised coordinates.

    StatsBomb uses a 120×80 pitch with ``(0, 0)`` in the top‑left corner.  The
    conversion to the Ush Analytics normalised system is::

        x_norm = x / 120
        y_norm = y / 80

    Parameters
    ----------
    statsbomb_df:
        DataFrame with ``x`` and ``y`` columns following the StatsBomb
        convention.
    attack_direction:
        ``"ltr"`` if the team attacks left‑to‑right (default) or ``"rtl"`` if
        it attacks right‑to‑left.  When ``"rtl"`` is passed the ``x`` coordinate
        is reflected using ``1 - x_norm``.

    Returns
    -------
    DataFrame
        Copy of ``statsbomb_df`` with additional ``x_norm`` and ``y_norm``
        columns.
    """

    df = statsbomb_df.copy()
    df["x_norm"] = df["x"] / 120
    df["y_norm"] = df["y"] / 80
    if attack_direction == "rtl":
        df["x_norm"] = 1 - df["x_norm"]
    return df


def to_ush_coords_wyscout(
    wyscout_df: pd.DataFrame, attack_direction: str = "ltr"
) -> pd.DataFrame:
    """Return Wyscout data with normalised coordinates.

    Wyscout coordinates are percentages on a 100×100 grid with the origin in
    the top‑left corner.  The conversion is::

        x_norm = x / 100
        y_norm = y / 100

    Parameters
    ----------
    wyscout_df:
        DataFrame with ``x`` and ``y`` columns following the Wyscout
        convention.
    attack_direction:
        ``"ltr"`` if the team attacks left‑to‑right (default) or ``"rtl"`` if
        it attacks right‑to‑left.  When ``"rtl"`` is passed the ``x`` coordinate
        is reflected using ``1 - x_norm``.

    Returns
    -------
    DataFrame
        Copy of ``wyscout_df`` with additional ``x_norm`` and ``y_norm``
        columns.
    """

    df = wyscout_df.copy()
    df["x_norm"] = df["x"] / 100
    df["y_norm"] = df["y"] / 100
    if attack_direction == "rtl":
        df["x_norm"] = 1 - df["x_norm"]
    return df


def to_ush_coords_opta(
    opta_df: pd.DataFrame, attack_direction: str = "ltr"
) -> pd.DataFrame:
    """Return Opta data with normalised coordinates.

    Opta represents positions on a 100×100 grid with ``(0, 0)`` in the
    bottom‑left corner.  To convert to the Ush Analytics system the vertical
    axis is flipped so that ``y`` increases from top to bottom::

        x_norm = x / 100
        y_norm = 1 - (y / 100)

    Parameters
    ----------
    opta_df:
        DataFrame with ``x`` and ``y`` columns following the Opta convention.
    attack_direction:
        ``"ltr"`` if the team attacks left‑to‑right (default) or ``"rtl"`` if
        it attacks right‑to‑left.  When ``"rtl"`` is passed the ``x`` coordinate
        is reflected using ``1 - x_norm``.

    Returns
    -------
    DataFrame
        Copy of ``opta_df`` with additional ``x_norm`` and ``y_norm`` columns.
    """

    df = opta_df.copy()
    df["x_norm"] = df["x"] / 100
    df["y_norm"] = 1 - (df["y"] / 100)
    if attack_direction == "rtl":
        df["x_norm"] = 1 - df["x_norm"]
    return df
