import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from run_all_pro import run_pipeline

def test_run_pipeline(tmp_path):
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
            "is_shot": 0,
            "is_goal": 0,
            "x": 50,
            "y": 40,
            "xg": 0.0,
            "minute": 5,
            "event_type": "Pass",
            "is_pass": 1,
            "player": "A1",
            "receiver": "A2",
            "end_x": 60,
            "end_y": 50,
            "is_def_action": 0,
        },
        {
            "match_id": 1,
            "team": "Away",
            "is_shot": 0,
            "is_goal": 0,
            "x": 52,
            "y": 42,
            "xg": 0.0,
            "minute": 15,
            "event_type": "Pass",
            "is_pass": 1,
            "player": "A1",
            "receiver": "A2",
            "end_x": 62,
            "end_y": 52,
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
            "away_goals": 0,
        }
    ])
    events_path = tmp_path / "events.csv"
    matches_path = tmp_path / "matches.csv"
    events.to_csv(events_path, index=False)
    matches.to_csv(matches_path, index=False)

    out_dir = tmp_path / "out"
    results = run_pipeline(events_path, matches_path, out_dir, match_id=1)
    assert len(results) == 1
    info = results[0]
    assert info["match_id"] == 1
    assert Path(info["report"]).exists()
