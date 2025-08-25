import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from ush_report import render_html_report_pro


def test_render_html_report_pro_creates_files(tmp_path):
    events = pd.DataFrame([
        {
            "team": "Home",
            "is_shot": 1,
            "is_goal": 0,
            "x": 100,
            "y": 40,
            "xg": 0.1,
            "minute": 10,
            "event_type": "Shot",
            "is_pass": 0,
            "player": "H1",
            "receiver": "",
            "end_x": None,
            "end_y": None,
            "is_def_action": 0,
        },
        {
            "team": "Away",
            "is_shot": 1,
            "is_goal": 1,
            "x": 80,
            "y": 30,
            "xg": 0.2,
            "minute": 20,
            "event_type": "Shot",
            "is_pass": 0,
            "player": "A1",
            "receiver": "",
            "end_x": None,
            "end_y": None,
            "is_def_action": 0,
        },
        {
            "team": "Away",
            "is_shot": 0,
            "is_goal": 0,
            "x": 50,
            "y": 50,
            "xg": 0.0,
            "minute": 30,
            "event_type": "Pass",
            "is_pass": 1,
            "player": "A1",
            "receiver": "A2",
            "end_x": 60,
            "end_y": 60,
            "is_def_action": 0,
        },
        {
            "team": "Away",
            "is_shot": 0,
            "is_goal": 0,
            "x": 52,
            "y": 52,
            "xg": 0.0,
            "minute": 35,
            "event_type": "Pass",
            "is_pass": 1,
            "player": "A1",
            "receiver": "A2",
            "end_x": 62,
            "end_y": 62,
            "is_def_action": 0,
        },
        {
            "team": "Home",
            "is_shot": 0,
            "is_goal": 0,
            "x": 40,
            "y": 60,
            "xg": 0.0,
            "minute": 44,
            "event_type": "Tackle",
            "is_pass": 0,
            "player": "H2",
            "receiver": "",
            "end_x": None,
            "end_y": None,
            "is_def_action": 1,
        },
    ])

    matches = pd.DataFrame([
        {
            "competition": "Friendly",
            "date": "2023-01-01",
            "venue_city": "City",
            "home_team": "Home",
            "away_team": "Away",
            "home_goals": 0,
            "away_goals": 1,
        }
    ])

    events_path = tmp_path / "events.csv"
    matches_path = tmp_path / "matches.csv"
    events.to_csv(events_path, index=False)
    matches.to_csv(matches_path, index=False)

    events = pd.read_csv(events_path)
    matches = pd.read_csv(matches_path)

    meta = matches.iloc[0].to_dict()
    teams = [meta.get("home_team"), meta.get("away_team")]

    shots = events[events["is_shot"] == 1]
    kpis = {}
    for t in teams:
        sub = shots[shots["team"] == t]
        passes = events[(events["team"] == t) & (events["is_pass"] == 1)]
        kpis[t] = {
            "shots": int(len(sub)),
            "goals": int(sub["is_goal"].sum()),
            "xg": float(sub["xg"].sum()),
            "xt": float(passes.shape[0]),
        }

    def ppda(df, team):
        req = {"is_pass", "is_def_action", "y"}
        if not req.issubset(df.columns):
            return None
        ymax = max(100.0, df["y"].max())
        yn = df["y"] / ymax * 100.0
        opp = ((df["team"] != team) & (df["is_pass"] == 1) & (yn >= 40)).sum()
        our = ((df["team"] == team) & (df["is_def_action"] == 1) & (yn >= 40)).sum()
        return round(float(opp) / float(our), 2) if our else None

    ppda_vals = {t: ppda(events, t) for t in teams}

    shotmap_path = tmp_path / "shotmap.png"
    xgrace_path = tmp_path / "xg_race.png"
    passnet_path = tmp_path / "pass_network.png"
    pressure_path = tmp_path / "pressure.png"
    for path, text in [
        (shotmap_path, "shotmap"),
        (xgrace_path, "xg race"),
        (passnet_path, "pass network"),
        (pressure_path, "pressure"),
    ]:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, text)
        fig.savefig(path)
        plt.close(fig)

    html_path = tmp_path / "report.html"
    logo_path = Path(__file__).resolve().parents[1] / "brand" / "ush_logo_dark.svg"

    render_html_report_pro(
        meta,
        teams,
        kpis,
        ppda_vals,
        shotmap_path,
        xgrace_path,
        passnet_path,
        pressure_path,
        logo_path,
        html_path,
    )

    assert shotmap_path.exists()
    assert xgrace_path.exists()
    assert passnet_path.exists()
    assert pressure_path.exists()
    assert html_path.exists()
