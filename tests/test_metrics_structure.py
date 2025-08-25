import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from metrics_structure import (
    pass_network,
    lane_carries,
    switches_throughballs_cutbacks,
    defensive_compactness,
    line_height,
)
from metrics_progression import load_xt_grid


def sample_events():
    """Create a minimal synthetic dataset for structural metrics."""
    return pd.DataFrame(
        [
            # Team A passes
            {"match_id": 1, "team": "A", "player": "A1", "receiver": "A2", "is_pass": 1, "is_carry": 0,
             "x": 10, "y": 10, "end_x": 30, "end_y": 10},
            {"match_id": 1, "team": "A", "player": "A2", "receiver": "A1", "is_pass": 1, "is_carry": 0,
             "x": 70, "y": 10, "end_x": 50, "end_y": 10},
            {"match_id": 1, "team": "A", "player": "A1", "receiver": "A3", "is_pass": 1, "is_carry": 0,
             "x": 50, "y": 10, "end_x": 70, "end_y": 70},
            # Team A carry
            {"match_id": 1, "team": "A", "player": "A3", "is_pass": 0, "is_carry": 1,
             "x": 60, "y": 70, "end_x": 80, "end_y": 70},
            # Team A defensive actions
            {"match_id": 1, "team": "A", "player": "A4", "is_pass": 0, "is_carry": 0, "is_def_action": 1,
             "x": 30, "y": 20},
            {"match_id": 1, "team": "A", "player": "A5", "is_pass": 0, "is_carry": 0, "is_def_action": 1,
             "x": 30, "y": 40},
            # Team B passes
            {"match_id": 1, "team": "B", "player": "B1", "receiver": "B2", "is_pass": 1, "is_carry": 0,
             "x": 90, "y": 70, "end_x": 70, "end_y": 70},
            {"match_id": 1, "team": "B", "player": "B2", "receiver": "B1", "is_pass": 1, "is_carry": 0,
             "x": 70, "y": 70, "end_x": 50, "end_y": 50},
            # Team B carry
            {"match_id": 1, "team": "B", "player": "B3", "is_pass": 0, "is_carry": 1,
             "x": 60, "y": 20, "end_x": 80, "end_y": 20},
            # Team B defensive actions
            {"match_id": 1, "team": "B", "player": "B4", "is_pass": 0, "is_carry": 0, "is_def_action": 1,
             "x": 90, "y": 60},
            {"match_id": 1, "team": "B", "player": "B5", "is_pass": 0, "is_carry": 0, "is_def_action": 1,
             "x": 90, "y": 40},
        ]
    )


def test_structure_metrics(tmp_path):
    events = sample_events()
    grid = load_xt_grid(Path(__file__).resolve().parents[1] / "data" / "xt_grid.csv")

    nodes, edges = pass_network(events, team="A", grid=grid)
    assert set(nodes.player) == {"A1", "A2", "A3"}
    a1 = nodes[nodes.player == "A1"].iloc[0]
    assert np.isclose(a1.x, (10 + 50 + 50) / 3)
    assert np.isclose(a1.y, 10)
    edge = edges[(edges.source == "A1") & (edges.target == "A3")].iloc[0]
    assert edge.weight == 1
    assert np.isclose(edge.xt_added, 0.11, atol=1e-6)

    carries = lane_carries(events)
    assert carries.loc[carries.team == "A", "right_carries"].iloc[0] == 1
    assert carries.loc[carries.team == "B", "left_carries"].iloc[0] == 1

    special = switches_throughballs_cutbacks(events)
    assert special.loc[special.team == "A", "switches"].iloc[0] == 1
    assert special.loc[special.team == "A", "throughballs"].iloc[0] == 1
    assert special.loc[special.team == "B", "cutbacks"].iloc[0] == 1

    compact = defensive_compactness(events)
    assert compact.loc[compact.team == "A", "defensive_compactness"].iloc[0] == 20
    assert compact.loc[compact.team == "B", "defensive_compactness"].iloc[0] == 20

    height = line_height(events)
    assert height.loc[height.team == "A", "line_height"].iloc[0] == 30
    assert height.loc[height.team == "B", "line_height"].iloc[0] == 90
