import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from metrics_progression import (
    progressive_pass,
    progressive_carry,
    load_xt_grid,
    lookup_xt,
    xt_lookup,
    xt_added,
    epv_simplified,
    xchain,
    xbuildup,
)


def sample_events():
    """Create a tiny synthetic event dataset for progression metrics."""
    return pd.DataFrame([
        # Team A actions
        {"match_id": 1, "team": "A", "minute": 10, "is_pass": 1, "is_carry": 0,
         "x": 30, "y": 40, "end_x": 60, "end_y": 40, "is_shot": 0, "xg": 0.0},
        {"match_id": 1, "team": "A", "minute": 15, "is_pass": 0, "is_carry": 1,
         "x": 60, "y": 40, "end_x": 80, "end_y": 40, "is_shot": 0, "xg": 0.0},
        {"match_id": 1, "team": "A", "minute": 20, "is_pass": 0, "is_carry": 0,
         "x": 100, "y": 40, "end_x": 100, "end_y": 40, "is_shot": 1, "xg": 0.3},
        {"match_id": 1, "team": "A", "minute": 90, "is_pass": 0, "is_carry": 0,
         "x": 0, "y": 0, "end_x": 0, "end_y": 0, "is_shot": 0, "xg": 0.0},
        # Team B actions
        {"match_id": 1, "team": "B", "minute": 30, "is_pass": 1, "is_carry": 0,
         "x": 40, "y": 40, "end_x": 45, "end_y": 40, "is_shot": 0, "xg": 0.0},
        {"match_id": 1, "team": "B", "minute": 40, "is_pass": 0, "is_carry": 0,
         "x": 90, "y": 40, "end_x": 90, "end_y": 40, "is_shot": 1, "xg": 0.2},
        {"match_id": 1, "team": "B", "minute": 90, "is_pass": 0, "is_carry": 0,
         "x": 0, "y": 0, "end_x": 0, "end_y": 0, "is_shot": 0, "xg": 0.0},
    ])


def test_progressive_actions_and_xt(tmp_path):
    events = sample_events()
    grid = load_xt_grid(Path(__file__).resolve().parents[1] / "data" / "xt_grid.csv")

    prog_pass = progressive_pass(events)
    assert prog_pass.loc[prog_pass.team == "A", "progressive_passes_per90"].iloc[0] == 1
    assert prog_pass.loc[prog_pass.team == "B", "progressive_passes_per90"].iloc[0] == 0

    prog_carry = progressive_carry(events)
    assert prog_carry.loc[prog_carry.team == "A", "progressive_carries_per90"].iloc[0] == 1
    assert prog_carry.loc[prog_carry.team == "B", "progressive_carries_per90"].iloc[0] == 0

    # xT lookups
    xt_df = xt_lookup(events, grid)
    a_pass = xt_df.iloc[0]
    assert pytest.approx(a_pass.start_xt, 1e-6) == 0.04
    assert pytest.approx(a_pass.end_xt, 1e-6) == 0.14

    xta = xt_added(events, grid)
    assert pytest.approx(xta.loc[xta.team == "A", "xt_added"].iloc[0], 1e-6) == 0.25
    assert pytest.approx(xta.loc[xta.team == "B", "xt_added"].iloc[0], 1e-6) == 0.0

    epv = epv_simplified(events, grid)
    assert pytest.approx(epv.loc[epv.team == "A", "epv_per90"].iloc[0], 1e-6) == 0.55
    assert pytest.approx(epv.loc[epv.team == "B", "epv_per90"].iloc[0], 1e-6) == 0.2

    xc = xchain(events)
    assert pytest.approx(xc.loc[xc.team == "A", "xg_chain_per90"].iloc[0], 1e-6) == 0.9
    assert pytest.approx(xc.loc[xc.team == "B", "xg_chain_per90"].iloc[0], 1e-6) == 0.4

    xb = xbuildup(events)
    assert pytest.approx(xb.loc[xb.team == "A", "xg_buildup_per90"].iloc[0], 1e-6) == 0.3
    assert pytest.approx(xb.loc[xb.team == "B", "xg_buildup_per90"].iloc[0], 1e-6) == 0.0
