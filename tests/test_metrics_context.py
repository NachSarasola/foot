import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from metrics_context import (
    score_state,
    phase_of_play,
    aggregate_by_score_state,
    aggregate_by_phase,
    aggregate_context,
)


def sample_events():
    """Return a small dataset for context metric tests."""
    return pd.DataFrame(
        [
            {"team": "A", "period": 1, "minute": 1, "is_goal": 0, "x": 10, "y": 50, "event_type": "pass", "xg": 0.1},
            {"team": "B", "period": 1, "minute": 2, "is_goal": 1, "x": 90, "y": 50, "event_type": "shot", "xg": 0.2},
            {"team": "A", "period": 1, "minute": 3, "is_goal": 0, "x": 50, "y": 40, "event_type": "pass", "xg": 0.05},
            {"team": "B", "period": 1, "minute": 4, "is_goal": 0, "x": 20, "y": 30, "event_type": "pass", "xg": 0.0},
            {"team": "A", "period": 2, "minute": 5, "is_goal": 1, "x": 95, "y": 60, "event_type": "shot", "xg": 0.3},
            {"team": "B", "period": 2, "minute": 6, "is_goal": 0, "x": 60, "y": 50, "event_type": "pass", "xg": 0.1},
        ]
    )


def test_state_and_phase_classification():
    events = sample_events()

    with_state = score_state(events)
    assert list(with_state["score_state"]) == [
        "level",
        "level",
        "behind",
        "ahead",
        "behind",
        "level",
    ]

    with_phase = phase_of_play(events)
    assert list(with_phase["phase"]) == [
        "build_up",
        "finalization",
        "progression",
        "build_up",
        "finalization",
        "progression",
    ]


def test_context_aggregations():
    events = sample_events()

    agg_state = aggregate_by_score_state(events, ["xg"])
    a_behind = agg_state.loc[(agg_state.team == "A") & (agg_state.score_state == "behind"), "xg"].iloc[0]
    b_level = agg_state.loc[(agg_state.team == "B") & (agg_state.score_state == "level"), "xg"].iloc[0]
    assert pytest.approx(a_behind, 1e-6) == pytest.approx(0.35, 1e-6)
    assert pytest.approx(b_level, 1e-6) == pytest.approx(0.3, 1e-6)

    agg_phase = aggregate_by_phase(events, ["xg"])
    a_final = agg_phase.loc[(agg_phase.team == "A") & (agg_phase.phase == "finalization"), "xg"].iloc[0]
    b_build = agg_phase.loc[(agg_phase.team == "B") & (agg_phase.phase == "build_up"), "xg"].iloc[0]
    assert pytest.approx(a_final, 1e-6) == pytest.approx(0.3, 1e-6)
    assert b_build == 0.0

    agg_all = aggregate_context(events, ["xg"])
    row = agg_all.loc[
        (agg_all.team == "A") & (agg_all.period == 1) & (agg_all.score_state == "behind") & (agg_all.phase == "progression"),
        "xg",
    ].iloc[0]
    assert pytest.approx(row, 1e-6) == pytest.approx(0.05, 1e-6)
