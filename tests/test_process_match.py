import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from run_all_pro import process_match


def test_process_match_missing_required_column(tmp_path):
    events = pd.DataFrame([
        {
            "match_id": 1,
            # "team" column intentionally omitted to trigger error
            "is_shot": 1,
            "is_goal": 0,
            "x": 100,
            "y": 40,
            "xg": 0.1,
            "minute": 10,
            "is_pass": 0,
            "is_def_action": 0,
            "end_x": None,
            "end_y": None,
        }
    ])

    meta = {
        "home_team": "Home",
        "away_team": "Away",
        "home_goals": 0,
        "away_goals": 0,
        "date": "2023-01-01",
    }

    brand_path = Path(__file__).resolve().parents[1] / "brand"

    with pytest.raises(ValueError, match="team"):
        process_match(events, meta, tmp_path, brand_path)

