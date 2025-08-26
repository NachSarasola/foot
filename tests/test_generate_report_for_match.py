import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from run_all_pro import generate_report_for_match


def test_generate_report_for_match_returns_paths(tmp_path):
    events = pd.DataFrame([
        {
            "match_id": 1,
            "team": "Home",
            "is_shot": 1,
            "is_goal": 0,
            "x": 100,
            "y": 40,
            "xg": 0.1,
            "minute": 10,
            "event_type": "Shot",
            "is_pass": 0,
            "player": "H1",
            "receiver": "",
            "end_x": None,
            "end_y": None,
            "is_def_action": 0,
        },
        {
            "match_id": 1,
            "team": "Away",
            "is_shot": 1,
            "is_goal": 1,
            "x": 80,
            "y": 30,
            "xg": 0.2,
            "minute": 20,
            "event_type": "Shot",
            "is_pass": 0,
            "player": "A1",
            "receiver": "",
            "end_x": None,
            "end_y": None,
            "is_def_action": 0,
        },
        {
            "match_id": 1,
            "team": "Away",
            "is_shot": 0,
            "is_goal": 0,
            "x": 50,
            "y": 50,
            "xg": 0.0,
            "minute": 30,
            "event_type": "Pass",
            "is_pass": 1,
            "player": "A1",
            "receiver": "A2",
            "end_x": 60,
            "end_y": 60,
            "is_def_action": 0,
        },
        {
            "match_id": 1,
            "team": "Away",
            "is_shot": 0,
            "is_goal": 0,
            "x": 52,
            "y": 52,
            "xg": 0.0,
            "minute": 35,
            "event_type": "Pass",
            "is_pass": 1,
            "player": "A1",
            "receiver": "A2",
            "end_x": 62,
            "end_y": 62,
            "is_def_action": 0,
        },
        {
            "match_id": 1,
            "team": "Home",
            "is_shot": 0,
            "is_goal": 0,
            "x": 40,
            "y": 60,
            "xg": 0.0,
            "minute": 44,
            "event_type": "Tackle",
            "is_pass": 0,
            "player": "H2",
            "receiver": "",
            "end_x": None,
            "end_y": None,
            "is_def_action": 1,
        },
    ])

    matches = pd.DataFrame([
        {
            "match_id": 1,
            "competition": "Friendly",
            "date": "2023-01-01",
            "venue_city": "City",
            "home_team": "Home",
            "away_team": "Away",
            "home_goals": 0,
            "away_goals": 1,
        }
    ])

    results = generate_report_for_match(events, matches, tmp_path, match_id=1)

    assert len(results) == 1
    info = results[0]
    assert info["match_id"] == 1
    expected = {
        "shotmap",
        "xg_race",
        "pass_network",
        "pressure_map",
        "shots_csv",
        "kpis_csv",
        "report",
    }
    assert expected.issubset(info.keys())
    for key in expected:
        assert Path(info[key]).exists(), f"missing {key}"


def test_generate_report_for_match_missing_column(tmp_path):
    events = pd.DataFrame([
        {
            "match_id": 1,
            "team": "Home",
            "is_shot": 1,
            "is_goal": 0,
            "x": 100,
            "y": 40,
            "xg": 0.1,
            "minute": 10,
            "event_type": "Shot",
            "is_pass": 0,
            "player": "H1",
            "receiver": "",
            "end_x": None,
            "end_y": None,
            "is_def_action": 0,
        },
    ])
    matches = pd.DataFrame([
        {
            "match_id": 1,
            "competition": "Friendly",
            "date": "2023-01-01",
            "venue_city": "City",
            "home_team": "Home",
            "away_team": "Away",
            "home_goals": 0,
            "away_goals": 1,
        }
    ])

    events = events.drop(columns=["receiver"])

    with pytest.raises(ValueError):
        generate_report_for_match(events, matches, tmp_path, match_id=1)
