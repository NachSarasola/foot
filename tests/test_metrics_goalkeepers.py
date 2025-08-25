import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from metrics_goalkeepers import (
    psxg_minus_ga,
    shot_stopping_pct,
    cross_claims,
    sweeper_actions,
    distribution_under_pressure,
)


def synthetic_goalkeeper_events():
    """Return a tiny goalkeeper dataset for testing."""
    return pd.DataFrame(
        [
            {"goalkeeper": "A", "shot_psxg": 0.8, "shot_outcome": "Goal", "under_pressure": 0, "pass_length": None},
            {"goalkeeper": "A", "shot_psxg": 0.3, "shot_outcome": "Saved", "under_pressure": 0, "pass_length": None},
            {"goalkeeper": "A", "shot_psxg": 0.2, "shot_outcome": "Goal", "under_pressure": 0, "pass_length": None},
            {"goalkeeper": "A", "shot_psxg": None, "shot_outcome": None, "under_pressure": 1, "pass_length": 25},
            {"goalkeeper": "A", "shot_psxg": None, "shot_outcome": None, "under_pressure": 1, "pass_length": 5},
            {"goalkeeper": "A", "shot_psxg": None, "shot_outcome": None, "under_pressure": 0, "pass_length": 30},
            {"goalkeeper": "B", "shot_psxg": 0.1, "shot_outcome": "Saved", "under_pressure": 0, "pass_length": None},
            {"goalkeeper": "B", "shot_psxg": 0.4, "shot_outcome": "Goal", "under_pressure": 0, "pass_length": None},
            {"goalkeeper": "B", "shot_psxg": 0.5, "shot_outcome": "Saved", "under_pressure": 0, "pass_length": None},
            {"goalkeeper": "B", "shot_psxg": None, "shot_outcome": None, "under_pressure": 1, "pass_length": 40},
            {"goalkeeper": "B", "shot_psxg": None, "shot_outcome": None, "under_pressure": 1, "pass_length": 50},
            {"goalkeeper": "B", "shot_psxg": None, "shot_outcome": None, "under_pressure": 0, "pass_length": 10},
        ]
    )


def test_goalkeeper_metrics():
    events = synthetic_goalkeeper_events()

    psxg_df = psxg_minus_ga(events)
    assert pytest.approx(psxg_df.loc[psxg_df.goalkeeper == "A", "psxg_minus_ga"].iloc[0], 1e-6) == pytest.approx(-0.7, 1e-6)
    assert pytest.approx(psxg_df.loc[psxg_df.goalkeeper == "B", "psxg_minus_ga"].iloc[0], 1e-6) == pytest.approx(0.0, 1e-6)

    stop_df = shot_stopping_pct(events)
    assert pytest.approx(stop_df.loc[stop_df.goalkeeper == "A", "shot_stopping_pct"].iloc[0], 1e-6) == pytest.approx(-0.5384615, 1e-6)
    assert stop_df.loc[stop_df.goalkeeper == "B", "shot_stopping_pct"].iloc[0] == 0

    claims_df = cross_claims(events)
    assert claims_df.loc[claims_df.goalkeeper == "A", "cross_claims"].iloc[0] == 2
    assert claims_df.loc[claims_df.goalkeeper == "B", "cross_claims"].iloc[0] == 2

    sweep_df = sweeper_actions(events)
    assert sweep_df.loc[sweep_df.goalkeeper == "A", "sweeper_actions"].iloc[0] == 1
    assert sweep_df.loc[sweep_df.goalkeeper == "B", "sweeper_actions"].iloc[0] == 2

    dist_df = distribution_under_pressure(events)
    assert dist_df.loc[dist_df.goalkeeper == "A", "distribution_under_pressure"].iloc[0] == pytest.approx(15)
    assert dist_df.loc[dist_df.goalkeeper == "B", "distribution_under_pressure"].iloc[0] == pytest.approx(45)
