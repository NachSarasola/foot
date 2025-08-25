# scripts/run_all_pro.py — Pipeline Pro (Ush)
# Ejecutar en VS Code con ▶️  (sin Jupyter). Requiere: pandas, numpy, matplotlib, mplsoccer, jinja2

# Auto-instalar si falta (opcional)
import sys, subprocess, importlib, argparse
for pkg in ["pandas", "numpy", "matplotlib", "mplsoccer", "jinja2"]:
    try:
        importlib.import_module(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", pkg])

from pathlib import Path
import pandas as pd, numpy as np, math
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# Estilo y helpers
from ush_style import (
    COLORS, set_ush_theme, add_grass_texture, scale_sizes, label,
    curved_edge, annotate_goals_scatter, annotate_goals_on_xg
)
from ush_report import render_html_report_pro


def xg_lite(row, xmax=120.0, ymax=80.0):
    dx = max(0.0, xmax - float(row.get('x', 0)))
    dy = float(row.get('y', 0)) - ymax / 2.0
    dist = (dx * dx + dy * dy) ** 0.5
    ang = np.arctan2(abs(dy), max(1e-6, dx))
    val = 1 / (1 + np.exp((dist - 18) / 4)) * (0.6 + 0.4 * (1 - ang / 1.57))
    return max(0.02, min(0.8, float(val)))


def ppda(df, team):
    req = {'is_pass', 'is_def_action', 'y'}
    if not req.issubset(df.columns):
        return None
    ymax = max(100.0, float(df['y'].max()) if 'y' in df.columns else 100.0)
    yn = df['y'] / ymax * 100.0
    opp = ((df['team'] != team) & (df['is_pass'] == 1) & (yn >= 40)).sum()
    our = ((df['team'] == team) & (df['is_def_action'] == 1) & (yn >= 40)).sum()
    return round(float(opp) / float(our), 2) if our else None


# ====== SHOT MAP — PRO ======
def draw_shot_map_pro(shots_df, teams, meta, out_path):
    if shots_df is None or shots_df.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.axis('off')
        ax.text(0.5, 0.5, "Sin tiros registrados", ha='center', va='center', fontsize=14)
        fig.savefig(out_path, dpi=220, bbox_inches='tight')
        plt.close(fig)
        return

    team_color = {teams[0]: COLORS['blue'], teams[1]: COLORS['cyan']}

    pitch = Pitch(pitch_type='statsbomb', pitch_color=COLORS['grass'], line_color=COLORS['fog'], linewidth=1)
    fig, ax = pitch.draw(figsize=(10, 7))
    add_grass_texture(ax, alpha=0.18)

    for t in teams:
        sub = shots_df.loc[shots_df['team'] == t, ['x', 'y', 'xg', 'is_goal', 'minute', 'player']].dropna(subset=['x', 'y'])
        if sub.empty:
            continue
        sizes = scale_sizes(sub['xg'].fillna(0.06) * 100, base=80, k=15, min_size=40, max_size=520)
        pitch.scatter(sub['x'], sub['y'], s=sizes, color=team_color[t], edgecolors='#ffffff',
                      linewidth=0.8, alpha=0.90, label=t, ax=ax, zorder=4)
        annotate_goals_scatter(ax, sub)

    leg = ax.legend(loc='lower left', frameon=False, fontsize=10)
    if leg is not None and leg.get_title() is not None:
        leg.get_title().set_color(COLORS['fog'])
    ax.set_title(f"Shot Map — {teams[1]} @ {teams[0]}  ({meta.get('date','')})", loc='left', pad=10, fontsize=13)

    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


# ====== XG RACE — PRO ======
def draw_xg_race_pro(shots_df, teams, meta, out_path):
    if shots_df is None or shots_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('off')
        ax.text(0.5, 0.5, "Sin tiros registrados", ha='center', va='center', fontsize=14)
        fig.savefig(out_path, dpi=220, bbox_inches='tight')
        plt.close(fig)
        return

    df = shots_df.copy()
    df['minute'] = pd.to_numeric(df.get('minute', 0), errors='coerce').fillna(0).astype(int)
    df['xg_val'] = pd.to_numeric(df.get('xg', 0), errors='coerce').fillna(0.06)
    minutes = np.arange(0, max(int(df['minute'].max()), 95) + 1)

    colors = {teams[0]: COLORS['blue'], teams[1]: COLORS['cyan']}
    series = {}
    for t in teams:
        by_min = (df[df['team'] == t].groupby('minute')['xg_val']
                  .sum().reindex(minutes, fill_value=0).cumsum())
        series[t] = by_min

    fig, ax = plt.subplots(figsize=(10, 5))
    for t in teams:
        ax.plot(minutes, series[t].values, lw=2.6, alpha=0.95, label=t, color=colors[t])

    lead = series[teams[1]] - series[teams[0]]
    ax.fill_between(minutes, series[teams[0]], series[teams[1]],
                    where=(lead >= 0), interpolate=True, color=COLORS['cyan'], alpha=0.08, zorder=0)
    ax.fill_between(minutes, series[teams[0]], series[teams[1]],
                    where=(lead < 0), interpolate=True, color=COLORS['blue'], alpha=0.08, zorder=0)

    annotate_goals_on_xg(ax, df, teams[0])
    annotate_goals_on_xg(ax, df, teams[1])
    ax.axvline(45, color='white', alpha=0.15, lw=1)
    ax.text(45, ax.get_ylim()[1] * 0.05, "HT", color=COLORS['fog'], ha='center',
            va='bottom', fontsize=9)

    ax.set_xlim(0, minutes[-1])
    ax.set_xlabel('Minuto')
    ax.set_ylabel('xG acumulado')
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    ax.set_title(f"xG Race — {teams[1]} @ {teams[0]}  ({meta.get('date','')})", loc='left', pad=10, fontsize=13)

    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


# ====== PASSING NETWORK — PRO ======
def draw_pass_network_pro(events_df, teams, meta, kpis, team_focus, out_path):
    df = events_df.copy()
    df_pass = df[(df.get('is_pass', 0) == 1) & (df.get('event_type') == 'Pass')]
    df_pass = df_pass[df_pass['team'] == team_focus].copy()

    has_receiver = ('receiver' in df_pass.columns) and df_pass['receiver'].notna().any() and (df_pass['receiver'] != '').any()
    if df_pass.empty or not has_receiver:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.axis('off')
        ax.text(0.5, 0.5, "Sin datos suficientes para red de pases", ha='center', va='center', fontsize=14, color=COLORS['fog'])
        fig.savefig(out_path, dpi=220, bbox_inches='tight')
        plt.close(fig)
        return

    starts = df_pass[['player', 'x', 'y']].rename(columns={'player': 'name'})
    recvs = df_pass[['receiver', 'end_x', 'end_y']].rename(columns={'receiver': 'name', 'end_x': 'x', 'end_y': 'y'})
    touches = pd.concat([starts, recvs], ignore_index=True).dropna(subset=['name', 'x', 'y'])
    locs = touches.groupby('name')[['x', 'y']].mean()
    touch_count = touches.groupby('name').size().rename('touches')

    links = (df_pass.groupby(['player', 'receiver']).size()
             .reset_index(name='count').sort_values('count', ascending=False))
    links = links[links['count'] >= 2]

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(13.5, 7.5))
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[0.33, 0.67], wspace=0.04, figure=fig)
    ax_info = fig.add_subplot(gs[0, 0])
    ax_pitch = fig.add_subplot(gs[0, 1])
    ax_info.set_facecolor(COLORS['navy'])
    ax_info.axis('off')

    ax_info.text(0.08, 0.92, "PASSING NETWORK", color="#cfe3ff", fontsize=13, fontweight="bold", transform=ax_info.transAxes)
    ax_info.text(0.08, 0.87, "CAPTURE DATA", color="#8aaad6", fontsize=9, transform=ax_info.transAxes)
    m_min = int(df_pass['minute'].min()) if 'minute' in df_pass.columns else 0
    m_max = int(df_pass['minute'].max()) if 'minute' in df_pass.columns else 90
    ax_info.text(
        0.08,
        0.82,
        f"PLAYER LOCATION: AVERAGE TOUCH POSITION\n{len(df_pass)} PASSES FROM {m_min:02d}' TO {m_max:02d}'",
        color="#cfe3ff",
        fontsize=9,
        linespacing=1.4,
        transform=ax_info.transAxes,
    )

    t1, t2 = teams[0], teams[1]
    xg_t1 = float(kpis.get(t1, {}).get('xg', float('nan')))
    xg_t2 = float(kpis.get(t2, {}).get('xg', float('nan')))
    ax_info.text(0.08, 0.64, f"{xg_t1:.1f}", color="#ffffff", fontsize=26, fontweight="bold", transform=ax_info.transAxes)
    ax_info.text(0.08, 0.60, f"xG {t1}", color="#8aaad6", fontsize=10, transform=ax_info.transAxes)
    ax_info.text(0.08, 0.52, f"{xg_t2:.1f}", color="#ffffff", fontsize=26, fontweight="bold", transform=ax_info.transAxes)
    ax_info.text(0.08, 0.48, f"xG {t2}", color="#8aaad6", fontsize=10, transform=ax_info.transAxes)
    ax_info.text(0.08, 0.18, "● TOU C H E S      ― PASSES", color="#8aaad6", fontsize=9, transform=ax_info.transAxes)

    pitch = Pitch(pitch_type='statsbomb', pitch_color=COLORS['grass'], line_color=COLORS['fog'], linewidth=1)
    pitch.draw(ax=ax_pitch)
    add_grass_texture(ax_pitch, alpha=0.18)

    for _, e in links.iterrows():
        a, b, w = e['player'], e['receiver'], int(e['count'])
        if a in locs.index and b in locs.index:
            xa, ya = float(locs.loc[a, 'x']), float(locs.loc[a, 'y'])
            xb, yb = float(locs.loc[b, 'x']), float(locs.loc[b, 'y'])
            curved_edge(ax_pitch, xa, ya, xb, yb, weight=w, color_main=COLORS['cyan'])

    sizes = scale_sizes(touch_count.reindex(locs.index).fillna(0).astype(int), base=160, k=25, min_size=160, max_size=900)
    pitch.scatter(locs['x'], locs['y'], s=sizes * 1.15, color='#ffffff', alpha=0.08, zorder=3, ax=ax_pitch)
    pitch.scatter(locs['x'], locs['y'], s=sizes, color=COLORS['cyan'], edgecolors=COLORS['fog'],
                  linewidth=0.8, alpha=0.95, zorder=4, ax=ax_pitch)
    for name, row in locs.iterrows():
        label(ax_pitch, row['x'], row['y'], str(name).replace('_', ' '))

    ax_pitch.set_title(f"Passing Network — {team_focus}", loc='left', color="#cfe3ff", fontsize=13, pad=10)
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main(events_path, matches_path, output_dir, team_focus=None):
    ROOT = Path(__file__).resolve().parents[1]
    BRAND = ROOT / 'brand'

    events_path = Path(events_path)
    matches_path = Path(matches_path)
    output_dir = Path(output_dir)

    report_dir = output_dir / 'report'
    img_dir = report_dir / 'img'
    pbi_dir = output_dir / 'powerbi_exports'
    for p in [report_dir, img_dir, pbi_dir]:
        p.mkdir(parents=True, exist_ok=True)

    set_ush_theme()

    events = pd.read_csv(events_path)
    matches = pd.read_csv(matches_path)
    meta = matches.iloc[0].to_dict()
    teams = [meta.get('home_team', 'Equipo A'), meta.get('away_team', 'Equipo B')]

    for c in ['is_shot', 'is_goal', 'is_pass', 'is_def_action']:
        if c in events.columns:
            events[c] = pd.to_numeric(events[c], errors='coerce').fillna(0).astype(int)
    for c in ['x', 'y', 'end_x', 'end_y', 'xg', 'minute']:
        if c in events.columns:
            events[c] = pd.to_numeric(events[c], errors='coerce')

    shots = events[events.get('is_shot', 0) == 1].copy()
    if 'xg' not in shots.columns or shots['xg'].isna().all():
        shots['xg'] = shots.apply(xg_lite, axis=1)

    kpis = {}
    for t in teams:
        sub = shots[shots['team'] == t]
        kpis[t] = {
            'shots': int(len(sub)),
            'goals': int(sub.get('is_goal', 0).sum()),
            'xg': float(round(sub.get('xg', 0).sum(), 2)),
        }

    ppda_vals = {t: ppda(events, t) for t in teams}

    shotmap_path = img_dir / 'shotmap.png'
    draw_shot_map_pro(shots, teams, meta, shotmap_path)
    print("Guardado:", shotmap_path)

    xgrace_path = img_dir / 'xg_race.png'
    draw_xg_race_pro(shots, teams, meta, xgrace_path)
    print("Guardado:", xgrace_path)

    if team_focus is None:
        team_focus = teams[1]
    passnet_path = img_dir / 'pass_network.png'
    draw_pass_network_pro(events, teams, meta, kpis, team_focus, passnet_path)
    print("Guardado:", passnet_path)

    shots[['match_id', 'team', 'minute', 'x', 'y', 'is_goal', 'xg']].to_csv(pbi_dir / 'shots.csv', index=False)
    pd.DataFrame([{'team': t, **kpis[t], 'ppda': ppda_vals[t]} for t in teams]).to_csv(pbi_dir / 'kpis.csv', index=False)

    report_path = report_dir / 'river_libertad_report.html'
    render_html_report_pro(meta, teams, kpis, ppda_vals, shotmap_path, xgrace_path,
                           passnet_path, BRAND / 'ush_logo_dark.svg', report_path)
    print("Listo →", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Ush Analytics pro report")
    parser.add_argument("--events", required=True, help="Path to events CSV")
    parser.add_argument("--matches", required=True, help="Path to matches CSV")
    parser.add_argument("--output", required=True, help="Output directory for generated files")
    parser.add_argument("--team-focus", help="Team to highlight in passing network")
    args = parser.parse_args()
    main(args.events, args.matches, args.output, args.team_focus)

