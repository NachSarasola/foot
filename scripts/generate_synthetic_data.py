"""Generate synthetic football event dataset.

This script creates a small self-contained dataset to be used in examples and
unit tests.  It writes ``events.csv``, ``matches.csv``, ``players.csv``,
``teams.csv`` and ``zones.csv`` to the ``data`` directory.
"""
from __future__ import annotations

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RNG = np.random.default_rng(0)


def generate_teams() -> pd.DataFrame:
    """Return a small set of synthetic teams."""
    teams = [
        {"team_id": i + 1, "name": f"Team {chr(65 + i)}", "city": f"City {i + 1}"}
        for i in range(4)
    ]
    return pd.DataFrame(teams)


def generate_players(teams: pd.DataFrame) -> pd.DataFrame:
    """Return players for each team."""
    positions = ["GK", "DF", "MF", "FW"]
    players: list[dict[str, object]] = []
    pid = 1
    for team_id in teams["team_id"]:
        for i in range(15):
            players.append(
                {
                    "player_id": pid,
                    "team_id": int(team_id),
                    "name": f"Player {pid}",
                    "position": positions[i % len(positions)],
                }
            )
            pid += 1
    return pd.DataFrame(players)


def generate_matches(teams: pd.DataFrame) -> pd.DataFrame:
    """Return a round robin schedule between teams."""
    team_ids = teams["team_id"].tolist()
    pairs = list(combinations(team_ids, 2))
    dates = pd.date_range("2023-07-01", periods=len(pairs), freq="7D")
    matches: list[dict[str, object]] = []
    for match_id, (date, (home, away)) in enumerate(zip(dates, pairs), 1):
        matches.append(
            {
                "match_id": match_id,
                "date": date.date(),
                "home_team_id": home,
                "away_team_id": away,
                "home_score": int(RNG.integers(0, 5)),
                "away_score": int(RNG.integers(0, 5)),
                "competition": "Synthetic League",
                "season": "2023",
            }
        )
    return pd.DataFrame(matches)


def generate_zones() -> pd.DataFrame:
    """Return simple pitch zones."""
    zones: list[dict[str, object]] = []
    zone_id = 1
    for (x_min, x_max) in [(0, 40), (40, 80), (80, 120)]:
        for (y_min, y_max) in [(0, 40), (40, 80)]:
            zones.append(
                {
                    "zone_id": zone_id,
                    "name": f"Zone {zone_id}",
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                }
            )
            zone_id += 1
    return pd.DataFrame(zones)


def generate_events(matches: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Return a synthetic event stream with coverage of all flags."""
    events: list[dict[str, object]] = []
    event_id = 1
    set_pieces = ["corner", "free_kick", "throw_in", "penalty"]
    for _, match in matches.iterrows():
        home = int(match["home_team_id"])
        away = int(match["away_team_id"])
        for _ in range(400):  # 6 matches -> 2400 events
            team_id = int(RNG.choice([home, away]))
            team_players = players.loc[players["team_id"] == team_id, "player_id"].values
            player_id = int(RNG.choice(team_players))
            minute = int(RNG.integers(0, 91))
            second = int(RNG.integers(0, 60))
            period = 1 if minute < 45 else 2
            event_type = RNG.choice(["pass", "carry", "shot"])
            x = float(RNG.uniform(0, 120))
            y = float(RNG.uniform(0, 80))
            end_x = float(np.clip(x + RNG.normal(0, 15), 0, 120))
            end_y = float(np.clip(y + RNG.normal(0, 15), 0, 80))
            outcome = RNG.choice(["complete", "incomplete"])
            is_shot = int(event_type == "shot")
            is_goal = int(is_shot and RNG.random() < 0.1)
            xg = float(RNG.uniform(0, 1)) if is_shot else 0.0
            pass_switch = int(event_type == "pass" and RNG.random() < 0.05)
            carry_progressive = int(event_type == "carry" and RNG.random() < 0.2)
            set_piece_type = ""
            if RNG.random() < 0.05:
                set_piece_type = str(RNG.choice(set_pieces))
            events.append(
                {
                    "event_id": event_id,
                    "match_id": int(match["match_id"]),
                    "team_id": team_id,
                    "player_id": player_id,
                    "period": period,
                    "minute": minute,
                    "second": second,
                    "event_type": event_type,
                    "x": x,
                    "y": y,
                    "end_x": end_x,
                    "end_y": end_y,
                    "outcome": outcome,
                    "is_shot": is_shot,
                    "is_goal": is_goal,
                    "xg": xg,
                    "pass_switch": pass_switch,
                    "carry_progressive": carry_progressive,
                    "set_piece_type": set_piece_type,
                }
            )
            event_id += 1
    # ensure flag coverage
    if not any(e["pass_switch"] == 1 for e in events):
        events[0]["pass_switch"] = 1
    if not any(e["carry_progressive"] == 1 for e in events):
        events[1]["carry_progressive"] = 1
    present = {e["set_piece_type"] for e in events}
    base = {
        "event_id": event_id,
        "match_id": int(matches.iloc[0]["match_id"]),
        "team_id": int(matches.iloc[0]["home_team_id"]),
        "player_id": int(
            players.loc[players["team_id"] == matches.iloc[0]["home_team_id"], "player_id"].iloc[0]
        ),
        "period": 1,
        "minute": 1,
        "second": 0,
        "event_type": "pass",
        "x": 50.0,
        "y": 40.0,
        "end_x": 60.0,
        "end_y": 40.0,
        "outcome": "complete",
        "is_shot": 0,
        "is_goal": 0,
        "xg": 0.0,
        "pass_switch": 0,
        "carry_progressive": 0,
        "set_piece_type": "",
    }
    for sp in set_pieces:
        if sp not in present:
            extra = base.copy()
            extra["event_id"] = event_id
            event_id += 1
            extra["set_piece_type"] = sp
            events.append(extra)
    return pd.DataFrame(events)


def main() -> None:
    """Generate dataset and write CSV files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    teams = generate_teams()
    players = generate_players(teams)
    matches = generate_matches(teams)
    zones = generate_zones()
    events = generate_events(matches, players)
    teams.to_csv(DATA_DIR / "teams.csv", index=False)
    players.to_csv(DATA_DIR / "players.csv", index=False)
    matches.to_csv(DATA_DIR / "matches.csv", index=False)
    zones.to_csv(DATA_DIR / "zones.csv", index=False)
    events.to_csv(DATA_DIR / "events.csv", index=False)


if __name__ == "__main__":
    main()
