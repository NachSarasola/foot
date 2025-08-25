import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from metrics_context import phase_of_play


def test_phase_assignment():
    df = pd.DataFrame(
        [
            {"x": 10, "y": 50, "event_type": "pass"},
            {"x": 60, "y": 40, "event_type": "pass"},
            {"x": 85, "y": 60, "event_type": "pass"},
            {"x": 30, "y": 70, "event_type": "shot"},
        ]
    )
    phases = phase_of_play(df)["phase"].tolist()
    assert phases == [
        "build_up",
        "progression",
        "finalization",
        "finalization",
    ]
