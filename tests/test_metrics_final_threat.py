import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from metrics_final_threat import (
    calculate_xg,
    calculate_xa,
    shot_creating_actions,
    deep_completions,
    box_entries,
    zone14_entries,
    passes_to_penalty_area,
    set_piece_xg,
)


def synthetic_events():
    """Generate a small synthetic event dataset for testing."""
    return pd.DataFrame([
        {"match_id": 1, "team": "A", "minute": 10, "is_pass": 1, "is_carry": 0,
         "is_shot": 0, "xg": 0, "xa": 0.2, "end_x": 105, "end_y": 40},
        {"match_id": 1, "team": "A", "minute": 11, "is_pass": 0, "is_carry": 0,
         "is_shot": 1, "xg": 0.2, "is_set_piece": 0},
        {"match_id": 1, "team": "A", "minute": 15, "is_pass": 0, "is_carry": 1,
         "is_shot": 0, "end_x": 104, "end_y": 50},
        {"match_id": 1, "team": "A", "minute": 20, "is_pass": 1, "is_carry": 0,
         "xa": 0.3, "is_shot": 0, "end_x": 90, "end_y": 40},
        {"match_id": 1, "team": "A", "minute": 21, "is_pass": 0, "is_carry": 0,
         "is_shot": 1, "xg": 0.3, "is_set_piece": 0},
        {"match_id": 1, "team": "A", "minute": 50, "is_pass": 0, "is_carry": 0,
         "is_shot": 1, "xg": 0.1, "is_set_piece": 1},
        {"match_id": 1, "team": "A", "minute": 90, "is_pass": 0, "is_carry": 0,
         "is_shot": 0, "xg": 0},
        {"match_id": 1, "team": "B", "minute": 30, "is_pass": 1, "is_carry": 0,
         "is_shot": 0, "xg": 0, "xa": 0.15, "end_x": 105, "end_y": 70},
        {"match_id": 1, "team": "B", "minute": 31, "is_pass": 0, "is_carry": 0,
         "is_shot": 1, "xg": 0.15, "is_set_piece": 0},
        {"match_id": 1, "team": "B", "minute": 60, "is_pass": 1, "is_carry": 0,
         "is_shot": 0, "xg": 0, "xa": 0.05, "end_x": 103, "end_y": 40},
        {"match_id": 1, "team": "B", "minute": 61, "is_pass": 0, "is_carry": 0,
         "is_shot": 1, "xg": 0.05, "is_set_piece": 0},
        {"match_id": 1, "team": "B", "minute": 70, "is_pass": 0, "is_carry": 0,
         "is_shot": 1, "xg": 0.2, "is_set_piece": 1},
        {"match_id": 1, "team": "B", "minute": 90, "is_pass": 0, "is_carry": 0,
         "is_shot": 0, "xg": 0},
    ])


def test_final_third_metrics():
    events = synthetic_events()

    xg = calculate_xg(events)
    assert pytest.approx(xg.loc[xg.team == "A", "xg_per90"].iloc[0], 1e-5) == 0.6
    assert pytest.approx(xg.loc[xg.team == "B", "xg_per90"].iloc[0], 1e-5) == 0.4

    xa = calculate_xa(events)
    assert pytest.approx(xa.loc[xa.team == "A", "xa_per90"].iloc[0], 1e-5) == 0.5
    assert pytest.approx(xa.loc[xa.team == "B", "xa_per90"].iloc[0], 1e-5) == 0.2

    sca = shot_creating_actions(events)
    assert sca.loc[sca.team == "A", "sca_per90"].iloc[0] == 3
    assert sca.loc[sca.team == "B", "sca_per90"].iloc[0] == 2

    deep = deep_completions(events)
    assert deep.loc[deep.team == "A", "deep_completions_per90"].iloc[0] == 0
    assert deep.loc[deep.team == "B", "deep_completions_per90"].iloc[0] == 1

    box = box_entries(events)
    assert box.loc[box.team == "A", "box_entries_per90"].iloc[0] == 2
    assert box.loc[box.team == "B", "box_entries_per90"].iloc[0] == 1

    z14 = zone14_entries(events)
    assert z14.loc[z14.team == "A", "zone14_entries_per90"].iloc[0] == 1
    assert z14.loc[z14.team == "B", "zone14_entries_per90"].iloc[0] == 0

    ppa = passes_to_penalty_area(events)
    assert ppa.loc[ppa.team == "A", "passes_to_penalty_area_per90"].iloc[0] == 1
    assert ppa.loc[ppa.team == "B", "passes_to_penalty_area_per90"].iloc[0] == 1

    spxg = set_piece_xg(events)
    assert pytest.approx(spxg.loc[spxg.team == "A", "set_piece_xg_per90"].iloc[0], 1e-5) == 0.1
    assert pytest.approx(spxg.loc[spxg.team == "B", "set_piece_xg_per90"].iloc[0], 1e-5) == 0.2
