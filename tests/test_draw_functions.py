import sys
from pathlib import Path
import hashlib
import pandas as pd

# Ensure scripts directory on path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'scripts'))

from run_all_pro import draw_shot_map_pro, draw_xg_race_pro, draw_pass_network_pro
from ush_style import set_ush_theme


def _sample_shots():
    return pd.DataFrame([
        {
            'team': 'Home',
            'is_shot': 1,
            'is_goal': 0,
            'x': 100,
            'y': 40,
            'xg': 0.1,
            'minute': 10,
        },
        {
            'team': 'Away',
            'is_shot': 1,
            'is_goal': 1,
            'x': 80,
            'y': 30,
            'xg': 0.2,
            'minute': 20,
        },
    ])


def _sample_events():
    return pd.DataFrame([
        {
            'team': 'Away',
            'is_pass': 1,
            'event_type': 'Pass',
            'player': 'A1',
            'receiver': 'A2',
            'x': 50,
            'y': 40,
            'end_x': 80,
            'end_y': 50,
            'minute': 5,
        },
        {
            'team': 'Away',
            'is_pass': 1,
            'event_type': 'Pass',
            'player': 'A1',
            'receiver': 'A2',
            'x': 55,
            'y': 45,
            'end_x': 85,
            'end_y': 55,
            'minute': 15,
        },
    ])


def _meta():
    return {'home_goals': 0, 'away_goals': 1, 'date': '2023-01-01'}


def _kpis():
    return {'Home': {'xg': 0.1}, 'Away': {'xg': 0.2}}


def _not_empty(path: Path) -> bool:
    empty_hash = hashlib.md5(b'').hexdigest()
    return hashlib.md5(path.read_bytes()).hexdigest() != empty_hash


def test_draw_shot_map_pro_creates_image(tmp_path):
    set_ush_theme()
    shots = _sample_shots()
    teams = ['Home', 'Away']
    meta = _meta()
    out = tmp_path / 'shotmap.png'
    draw_shot_map_pro(shots, teams, meta, out)
    assert out.exists()
    assert _not_empty(out)


def test_draw_xg_race_pro_creates_image(tmp_path):
    set_ush_theme()
    shots = _sample_shots()
    teams = ['Home', 'Away']
    meta = _meta()
    out = tmp_path / 'xg_race.png'
    draw_xg_race_pro(shots, teams, meta, out)
    assert out.exists()
    assert _not_empty(out)


def test_draw_pass_network_pro_creates_image(tmp_path):
    set_ush_theme()
    events = _sample_events()
    teams = ['Home', 'Away']
    meta = _meta()
    kpis = _kpis()
    out = tmp_path / 'passnet.png'
    draw_pass_network_pro(events, teams, meta, kpis, 'Away', out)
    assert out.exists()
    assert _not_empty(out)
