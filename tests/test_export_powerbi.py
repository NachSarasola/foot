import sys
from pathlib import Path

import pandas as pd

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from export_powerbi import export_powerbi

DATA = Path(__file__).resolve().parents[1] / "data"


def test_export_powerbi_generates_csvs(tmp_path):
    out_dir = tmp_path / "pbi"
    export_powerbi(
        events_csv=DATA / "events.csv",
        teams_csv=DATA / "teams.csv",
        players_csv=DATA / "players.csv",
        xt_grid_csv=DATA / "xt_grid.csv",
        output_dir=out_dir,
    )

    # Shots
    shots_path = out_dir / "shots.csv"
    assert shots_path.exists()
    shots = pd.read_csv(shots_path)
    assert list(shots.columns) == [
        "match_id",
        "team",
        "player",
        "minute",
        "x",
        "y",
        "is_goal",
        "xg",
    ]

    # KPIs
    kpis_path = out_dir / "kpis.csv"
    assert kpis_path.exists()
    kpis = pd.read_csv(kpis_path)
    assert list(kpis.columns) == ["team", "shots", "goals", "xg", "xt"]

    # xT events
    xt_events_path = out_dir / "xT_events.csv"
    assert xt_events_path.exists()
    xt_events = pd.read_csv(xt_events_path)
    assert list(xt_events.columns) == [
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

    # Regains
    regains_path = out_dir / "regains.csv"
    assert regains_path.exists()
    regains = pd.read_csv(regains_path)
    assert list(regains.columns) == [
        "match_id",
        "team",
        "player",
        "minute",
        "second",
        "x",
        "y",
    ]

    # Set pieces
    setpieces_path = out_dir / "setpieces.csv"
    assert setpieces_path.exists()
    setpieces = pd.read_csv(setpieces_path)
    assert list(setpieces.columns) == [
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

    # Player summary
    player_summary_path = out_dir / "player_summary.csv"
    assert player_summary_path.exists()
    player_summary = pd.read_csv(player_summary_path)
    assert list(player_summary.columns) == ["player", "team", "shots", "goals", "xg"]
