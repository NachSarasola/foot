import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from coord_norm import (  # noqa: E402
    to_ush_coords_statsbomb,
    to_ush_coords_wyscout,
    to_ush_coords_opta,
)


def test_statsbomb_ranges_and_reflection():
    df = pd.DataFrame({"x": [0, 60, 120], "y": [0, 40, 80]})
    ltr = to_ush_coords_statsbomb(df)
    assert ltr["x_norm"].between(0, 1).all()
    assert ltr["y_norm"].between(0, 1).all()

    rtl = to_ush_coords_statsbomb(df, attack_direction="rtl")
    assert np.allclose(rtl["x_norm"], 1 - ltr["x_norm"])


def test_wyscout_ranges_and_reflection():
    df = pd.DataFrame({"x": [0, 50, 100], "y": [0, 50, 100]})
    ltr = to_ush_coords_wyscout(df)
    assert ltr["x_norm"].between(0, 1).all()
    assert ltr["y_norm"].between(0, 1).all()

    rtl = to_ush_coords_wyscout(df, attack_direction="rtl")
    assert np.allclose(rtl["x_norm"], 1 - ltr["x_norm"])


def test_opta_ranges_and_reflection():
    df = pd.DataFrame({"x": [0, 50, 100], "y": [0, 50, 100]})
    ltr = to_ush_coords_opta(df)
    assert ltr["x_norm"].between(0, 1).all()
    assert ltr["y_norm"].between(0, 1).all()
    assert ltr.loc[0, "y_norm"] == pytest.approx(1.0)
    assert ltr.loc[2, "y_norm"] == pytest.approx(0.0)

    rtl = to_ush_coords_opta(df, attack_direction="rtl")
    assert np.allclose(rtl["x_norm"], 1 - ltr["x_norm"])
