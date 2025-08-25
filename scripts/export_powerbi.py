"""Export CSVs for Power BI dashboards.

This script reads match event data, computes basic KPIs along with
expected threat (xT) values and outputs several CSV files that can be
consumed by a Power BI model.  It is intentionally lightweight and
suitable for the synthetic datasets used in the tests.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from metrics_progression import load_xt_grid, xt_lookup


def export_powerbi(
    events_csv: Path,
    teams_csv: Path,
    players_csv: Path,
    xt_grid_csv: Path,
    output_dir: Path,
) -> None:
    """Generate CSV exports required by the Power BI model.

    Parameters
    ----------
    events_csv : Path
        Path to the raw events CSV file.
    teams_csv : Path
        Path to the teams reference CSV.
    players_csv : Path
        Path to the players reference CSV.
    xt_grid_csv : Path
        Path to the open‑xT grid CSV (8×12 cells).
    output_dir : Path
        Directory where the CSV exports will be written.
    """
    events = pd.read_csv(events_csv)
    teams = pd.read_csv(teams_csv)[["team_id", "name"]].rename(columns={"name": "team"})
    players = (
        pd.read_csv(players_csv)[["player_id", "name"]]
        .rename(columns={"name": "player"})
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Map team and player names onto events
    events = events.merge(teams, on="team_id", how="left")
    events = events.merge(players, on="player_id", how="left")

    # Derive flags for passes and carries to make use of xt_lookup
    event_type = events["event_type"].str.lower()
    events["is_pass"] = (event_type == "pass").astype(int)
    events["is_carry"] = (event_type == "carry").astype(int)

    grid = load_xt_grid(xt_grid_csv)
    events_xt = xt_lookup(events, grid)
    events_xt["xt_added"] = events_xt["end_xt"] - events_xt["start_xt"]

    # ------------------------------------------------------------------
    # Shots and basic KPIs
    # ------------------------------------------------------------------
    shots = events_xt[events_xt.get("is_shot", 0) == 1][
        ["match_id", "team", "player", "minute", "x", "y", "is_goal", "xg"]
    ]
    shots.to_csv(output_dir / "shots.csv", index=False)

    kpis = (
        shots.groupby("team")
        .agg(shots=("xg", "size"), goals=("is_goal", "sum"), xg=("xg", "sum"))
        .reset_index()
    )
    xt_team = (
        events_xt.groupby("team")["xt_added"].sum().reset_index().rename(columns={"xt_added": "xt"})
    )
    kpis = kpis.merge(xt_team, on="team", how="left")
    kpis.to_csv(output_dir / "kpis.csv", index=False)

    # ------------------------------------------------------------------
    # xT events, regains and set pieces
    # ------------------------------------------------------------------
    xt_events = events_xt[(events_xt["is_pass"] == 1) | (events_xt["is_carry"] == 1)][
        [
            "match_id",
            "team",
            "player",
            "event_type",
            "x",
            "y",
            "end_x",
            "end_y",
            "start_xt",
            "end_xt",
            "xt_added",
        ]
    ]
    xt_events.to_csv(output_dir / "xT_events.csv", index=False)

    regains = events_xt[events_xt.get("regain", 0) == 1][
        ["match_id", "team", "player", "minute", "second", "x", "y"]
    ]
    regains.to_csv(output_dir / "regains.csv", index=False)

    setpieces = events_xt[
        events_xt["set_piece_type"].notna()
        & (events_xt["set_piece_type"].astype(str).str.strip() != "")
    ][
        [
            "match_id",
            "team",
            "player",
            "minute",
            "second",
            "event_type",
            "set_piece_type",
            "x",
            "y",
        ]
    ]
    setpieces.to_csv(output_dir / "setpieces.csv", index=False)

    # ------------------------------------------------------------------
    # Player summary
    # ------------------------------------------------------------------
    player_summary = (
        shots.groupby(["player", "team"])
        .agg(shots=("xg", "size"), goals=("is_goal", "sum"), xg=("xg", "sum"))
        .reset_index()
    )
    player_summary.to_csv(output_dir / "player_summary.csv", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    """Return an argument parser for the CLI."""
    parser = argparse.ArgumentParser(description="Export Power BI CSVs")
    parser.add_argument("--events", type=Path, default=Path("data/events.csv"))
    parser.add_argument("--teams", type=Path, default=Path("data/teams.csv"))
    parser.add_argument("--players", type=Path, default=Path("data/players.csv"))
    parser.add_argument("--xt-grid", dest="xt_grid", type=Path, default=Path("data/xt_grid.csv"))
    parser.add_argument("--output", type=Path, default=Path("powerbi_exports"))
    return parser


def main() -> None:
    """Run the export process using command line arguments."""
    parser = build_arg_parser()
    args = parser.parse_args()
    export_powerbi(
        events_csv=args.events,
        teams_csv=args.teams,
        players_csv=args.players,
        xt_grid_csv=args.xt_grid,
        output_dir=args.output,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
