import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
import metrics_context


def valid_events():
    return pd.DataFrame(
        [
            {
                "team": "A",
                "period": 1,
                "is_goal": 1,
                "is_pass": 1,
                "is_def_action": 1,
                "xt_added": 0.2,
            },
            {
                "team": "B",
                "period": 1,
                "is_goal": 1,
                "is_pass": 1,
                "is_def_action": 1,
                "xt_added": 0.3,
            },
        ]
    )


def test_validate_events_passes(monkeypatch):
    events = valid_events()
    matches = pd.DataFrame([{"home_score": 1, "away_score": 1}])
    monkeypatch.setattr(metrics_context.pd, "read_csv", lambda *_args, **_kwargs: matches)
    metrics_context.validate_events(events)


def test_validate_events_goal_mismatch(monkeypatch):
    events = valid_events()
    matches = pd.DataFrame([{"home_score": 2, "away_score": 1}])
    monkeypatch.setattr(metrics_context.pd, "read_csv", lambda *_args, **_kwargs: matches)
    with pytest.raises(ValueError, match="goals"):
        metrics_context.validate_events(events)


def test_validate_events_invalid_ppda(monkeypatch):
    events = pd.DataFrame(
        [
            {
                "team": "A",
                "period": 1,
                "is_goal": 0,
                "is_pass": 1,
                "is_def_action": 1,
                "xt_added": 0.1,
            },
            {
                "team": "B",
                "period": 1,
                "is_goal": 0,
                "is_pass": 1,
                "is_def_action": 0,
                "xt_added": 0.2,
            },
        ]
    )
    matches = pd.DataFrame([{"home_score": 0, "away_score": 0}])
    monkeypatch.setattr(metrics_context.pd, "read_csv", lambda *_args, **_kwargs: matches)
    with pytest.raises(ValueError, match="PPDA"):
        metrics_context.validate_events(events)


def test_validate_events_negative_xt(monkeypatch):
    events = pd.DataFrame(
        [
            {
                "team": "A",
                "period": 1,
                "is_goal": 0,
                "is_pass": 1,
                "is_def_action": 1,
                "xt_added": -0.1,
            },
            {
                "team": "B",
                "period": 1,
                "is_goal": 0,
                "is_pass": 1,
                "is_def_action": 1,
                "xt_added": 0.0,
            },
        ]
    )
    matches = pd.DataFrame([{"home_score": 0, "away_score": 0}])
    monkeypatch.setattr(metrics_context.pd, "read_csv", lambda *_args, **_kwargs: matches)
    with pytest.raises(ValueError, match="xt_added"):
        metrics_context.validate_events(events)
