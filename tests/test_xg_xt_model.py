import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from run_all_pro import xg_model, xt_lite, calculate_xt


def test_xg_model_range_and_order():
    central_close = xg_model({"x": 110, "y": 40})
    distant = xg_model({"x": 60, "y": 40})
    assert 0 < distant < central_close < 1
    assert central_close > 0.5
    assert distant < 0.05


def test_xt_lite_and_calculate_xt():
    df = pd.DataFrame([
        {"team": "A", "is_pass": 1, "x": 80, "y": 40, "end_x": 100, "end_y": 40},
        {"team": "A", "is_pass": 1, "x": 100, "y": 40, "end_x": 80, "end_y": 40},
        {"team": "B", "is_pass": 1, "x": 60, "y": 40, "end_x": 61, "end_y": 40},
    ])
    xt_values = df.apply(xt_lite, axis=1)
    assert xt_values.iloc[0] > 0  # forward pass increases threat
    assert xt_values.iloc[1] == 0  # backward pass does not increase threat
    totals = calculate_xt(df, ["A", "B"])
    assert totals["A"] > 0
    assert 0 <= totals["B"] < 0.1
