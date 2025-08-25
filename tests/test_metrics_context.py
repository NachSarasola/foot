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
            {"team": "A", "period": 1, "minute": 1, "is_goal": 0, "in_possession": 1, "is_set_piece": 0, "xg": 0.1},
            {"team": "B", "period": 1, "minute": 2, "is_goal": 1, "in_possession": 1, "is_set_piece": 0, "xg": 0.2},
            {"team": "A", "period": 1, "minute": 3, "is_goal": 0, "in_possession": 1, "is_set_piece": 1, "xg": 0.05},
            {"team": "B", "period": 1, "minute": 4, "is_goal": 0, "in_possession": 0, "is_set_piece": 0, "xg": 0.0},
            {"team": "A", "period": 2, "minute": 5, "is_goal": 1, "in_possession": 1, "is_set_piece": 0, "xg": 0.3},
            {"team": "B", "period": 2, "minute": 6, "is_goal": 0, "in_possession": 1, "is_set_piece": 0, "xg": 0.1},
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
        "attack",
        "attack",
        "set_piece",
        "defence",
        "attack",
        "attack",
    ]


def test_context_aggregations():
    events = sample_events()

    agg_state = aggregate_by_score_state(events, ["xg"])
    a_behind = agg_state.loc[(agg_state.team == "A") & (agg_state.score_state == "behind"), "xg"].iloc[0]
    b_level = agg_state.loc[(agg_state.team == "B") & (agg_state.score_state == "level"), "xg"].iloc[0]
    assert pytest.approx(a_behind, 1e-6) == pytest.approx(0.35, 1e-6)
    assert pytest.approx(b_level, 1e-6) == pytest.approx(0.3, 1e-6)

    agg_phase = aggregate_by_phase(events, ["xg"])
    a_attack = agg_phase.loc[(agg_phase.team == "A") & (agg_phase.phase == "attack"), "xg"].iloc[0]
    b_def = agg_phase.loc[(agg_phase.team == "B") & (agg_phase.phase == "defence"), "xg"].iloc[0]
    assert pytest.approx(a_attack, 1e-6) == pytest.approx(0.4, 1e-6)
    assert b_def == 0.0

    agg_all = aggregate_context(events, ["xg"])
    row = agg_all.loc[
        (agg_all.team == "A") & (agg_all.period == 1) & (agg_all.score_state == "behind") & (agg_all.phase == "set_piece"),
        "xg",
    ].iloc[0]
    assert pytest.approx(row, 1e-6) == pytest.approx(0.05, 1e-6)
