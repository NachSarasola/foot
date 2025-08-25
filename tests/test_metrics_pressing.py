import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from metrics_pressing import (
    ppda,
    oppda,
    high_regains,
    counterpress_regains,
    high_turnovers,
    press_intensity,
    field_tilt,
)


def synthetic_events():
    """Return a tiny pressing dataset for testing."""
    return pd.DataFrame(
        [
            {"team": "A", "is_pass": 1, "is_def_action": 0, "turnover": 0, "recovery": 0, "tackle": 0, "block": 0, "y": 50, "period": 1, "possession_id": 1},
            {"team": "B", "is_pass": 0, "is_def_action": 1, "turnover": 0, "recovery": 0, "tackle": 1, "block": 0, "y": 50, "period": 1, "possession_id": 2},
            {"team": "B", "is_pass": 1, "is_def_action": 0, "turnover": 0, "recovery": 0, "tackle": 0, "block": 0, "y": 40, "period": 1, "possession_id": 2},
            {"team": "A", "is_pass": 0, "is_def_action": 0, "turnover": 1, "recovery": 0, "tackle": 0, "block": 0, "y": 65, "period": 1, "possession_id": 3},
            {"team": "B", "is_pass": 1, "is_def_action": 0, "turnover": 0, "recovery": 0, "tackle": 0, "block": 0, "y": 45, "period": 1, "possession_id": 4},
            {"team": "A", "is_pass": 0, "is_def_action": 0, "turnover": 0, "recovery": 1, "tackle": 0, "block": 0, "y": 70, "period": 1, "possession_id": 5},
            {"team": "B", "is_pass": 1, "is_def_action": 0, "turnover": 0, "recovery": 0, "tackle": 0, "block": 0, "y": 70, "period": 1, "possession_id": 6},
            {"team": "A", "is_pass": 0, "is_def_action": 1, "turnover": 0, "recovery": 0, "tackle": 1, "block": 0, "y": 70, "period": 1, "possession_id": 7},
            {"team": "A", "is_pass": 1, "is_def_action": 0, "turnover": 0, "recovery": 0, "tackle": 0, "block": 0, "y": 65, "period": 1, "possession_id": 7},
            {"team": "B", "is_pass": 0, "is_def_action": 0, "turnover": 1, "recovery": 0, "tackle": 0, "block": 0, "y": 70, "period": 1, "possession_id": 8},
            {"team": "A", "is_pass": 1, "is_def_action": 0, "turnover": 0, "recovery": 0, "tackle": 0, "block": 0, "y": 70, "period": 1, "possession_id": 9},
            {"team": "B", "is_pass": 1, "is_def_action": 0, "turnover": 0, "recovery": 0, "tackle": 0, "block": 0, "y": 50, "period": 2, "possession_id": 10},
            {"team": "A", "is_pass": 0, "is_def_action": 1, "turnover": 0, "recovery": 0, "tackle": 1, "block": 0, "y": 55, "period": 2, "possession_id": 11},
            {"team": "B", "is_pass": 1, "is_def_action": 0, "turnover": 0, "recovery": 0, "tackle": 0, "block": 0, "y": 70, "period": 2, "possession_id": 12},
            {"team": "A", "is_pass": 0, "is_def_action": 0, "turnover": 0, "recovery": 1, "tackle": 0, "block": 0, "y": 70, "period": 2, "possession_id": 13},
        ]
    )


def test_pressing_metrics():
    events = synthetic_events()

    ppda_df = ppda(events)
    assert ppda_df.loc[(ppda_df.team == "A") & (ppda_df.period == 1), "ppda"].iloc[0] == 3
    assert ppda_df.loc[(ppda_df.team == "B") & (ppda_df.period == 1), "ppda"].iloc[0] == 3

    oppda_df = oppda(events)
    assert oppda_df.loc[(oppda_df.team == "A") & (oppda_df.period == 1), "oppda"].iloc[0] == 3
    assert oppda_df.loc[(oppda_df.team == "B") & (oppda_df.period == 1), "oppda"].iloc[0] == 3

    regains = high_regains(events)
    assert regains.loc[(regains.team == "A") & (regains.period == 1), "high_regains"].iloc[0] == 2
    assert regains.loc[(regains.team == "B") & (regains.period == 1), "high_regains"].iloc[0] == 0

    counter = counterpress_regains(events)
    assert counter.loc[(counter.team == "A") & (counter.period == 1), "counterpress_regains"].iloc[0] == 1
    assert counter.loc[(counter.team == "B") & (counter.period == 1), "counterpress_regains"].iloc[0] == 0

    turnovers = high_turnovers(events)
    assert turnovers.loc[(turnovers.team == "A") & (turnovers.period == 1), "high_turnovers"].iloc[0] == 1
    assert turnovers.loc[(turnovers.team == "B") & (turnovers.period == 1), "high_turnovers"].iloc[0] == 1

    intensity = press_intensity(events)
    assert pytest.approx(intensity.loc[(intensity.team == "A") & (intensity.period == 1), "press_intensity"].iloc[0], 1e-6) == pytest.approx(1 / 3, 1e-6)
    assert intensity.loc[(intensity.team == "B") & (intensity.period == 1), "press_intensity"].iloc[0] == 0

    tilt = field_tilt(events)
    assert pytest.approx(tilt.loc[(tilt.team == "A") & (tilt.period == 1), "field_tilt"].iloc[0], 1e-6) == pytest.approx(2 / 3, 1e-6)
    assert pytest.approx(tilt.loc[(tilt.team == "B") & (tilt.period == 1), "field_tilt"].iloc[0], 1e-6) == pytest.approx(1 / 3, 1e-6)
    assert tilt.loc[(tilt.team == "A") & (tilt.period == 2), "field_tilt"].iloc[0] == 0
    assert tilt.loc[(tilt.team == "B") & (tilt.period == 2), "field_tilt"].iloc[0] == 1
