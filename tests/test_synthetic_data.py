import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def test_files_exist_and_nonempty():
    for name in ["events.csv", "matches.csv", "players.csv", "teams.csv", "zones.csv"]:
        path = DATA / name
        assert path.exists(), f"missing {name}"
        df = pd.read_csv(path)
        assert len(df) > 0, f"{name} empty"


def test_events_flags_and_ranges():
    events = pd.read_csv(DATA / "events.csv")
    assert len(events) >= 2000
    required = {"pass_switch", "carry_progressive", "set_piece_type"}
    assert required.issubset(events.columns)
    assert events["x"].between(0, 120).all()
    assert events["y"].between(0, 80).all()
    assert events["pass_switch"].isin([0, 1]).all()
    assert events["carry_progressive"].isin([0, 1]).all()
    assert events["pass_switch"].sum() > 0
    assert events["pass_switch"].sum() < len(events)
    assert events["carry_progressive"].sum() > 0
    assert events["carry_progressive"].sum() < len(events)
    set_types = {"corner", "free_kick", "throw_in", "penalty"}
    present = set(events["set_piece_type"].dropna().unique()) - {""}
    assert set_types.issubset(present)


def test_matches_players_teams_links():
    matches = pd.read_csv(DATA / "matches.csv")
    teams = pd.read_csv(DATA / "teams.csv")
    players = pd.read_csv(DATA / "players.csv")
    assert matches["home_team_id"].isin(teams["team_id"]).all()
    assert matches["away_team_id"].isin(teams["team_id"]).all()
    assert players["team_id"].isin(teams["team_id"]).all()
    assert matches["home_score"].ge(0).all()
    assert matches["away_score"].ge(0).all()


def test_zones_bounds():
    zones = pd.read_csv(DATA / "zones.csv")
    assert zones["x_min"].between(0, 120).all()
    assert zones["x_max"].between(0, 120).all()
    assert zones["y_min"].between(0, 80).all()
    assert zones["y_max"].between(0, 80).all()
    assert (zones["x_min"] < zones["x_max"]).all()
    assert (zones["y_min"] < zones["y_max"]).all()
