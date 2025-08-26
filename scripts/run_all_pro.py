# scripts/run_all_pro.py — Pipeline Pro (Ush)
# Ejecutar en VS Code con ▶️  (sin Jupyter). Requiere: pandas, numpy, matplotlib, mplsoccer, jinja2


# Dependencias
import argparse
from pathlib import Path
import pandas as pd, numpy as np, math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mplsoccer import Pitch

# Estilo y helpers
from ush_style import (
    COLORS, add_grass_texture, scale_sizes, label,
    curved_edge, annotate_goals_scatter, annotate_goals_on_xg,
    annotate_score, shot_marker_kwargs, shot_on_target_mask
)
from ush_report import render_html_report_pro

# Dimensiones de salida
WIDTH_PX = 1600
HEIGHT_PX = 1000
DPI = 240


def xg_model(row, xmax=120.0, ymax=80.0):
    """Calibrated xG model based on distance and shooting angle.

    Parameters
    ----------
    row : mapping
        Must contain ``x`` and ``y`` coordinates.
    xmax, ymax : float, optional
        Pitch dimensions used to normalize input.

    Returns
    -------
    float
        Expected goal probability between 0 and 1.
    """
    x = float(row.get("x", 0))
    y = float(row.get("y", 0))
    dx = xmax - x
    dy = abs(y - ymax / 2.0)
    dist = math.hypot(dx, dy)
    angle = math.atan2(7.32 / 2.0, dist)
    logit = -0.1135 + (-0.09 * dist) + (15.0 * angle)
    return 1.0 / (1.0 + math.exp(-logit))


def ppda(df, team):
    req = {'is_pass', 'is_def_action', 'y'}
    if not req.issubset(df.columns):
        return None
    ymax = max(100.0, float(df['y'].max()) if 'y' in df.columns else 100.0)
    yn = df['y'] / ymax * 100.0
    opp = ((df['team'] != team) & (df['is_pass'] == 1) & (yn >= 40)).sum()
    our = ((df['team'] == team) & (df['is_def_action'] == 1) & (yn >= 40)).sum()
    return round(float(opp) / float(our), 2) if our else None


def xt_lite(row):
    """Estimate xT gain of a single action using the calibrated xG model."""
    start = xg_model({"x": row.get("x", 0), "y": row.get("y", 0)})
    end = xg_model({"x": row.get("end_x", row.get("x", 0)), "y": row.get("end_y", row.get("y", 0))})
    return max(0.0, end - start)


def calculate_xt(df, teams):
    """Return total xT per team based on pass progression.

    The function relies on :func:`xg_model` to estimate the value of
    the starting and ending locations of each pass.
    """
    req = {'team', 'is_pass', 'x', 'y', 'end_x', 'end_y'}
    if not req.issubset(df.columns):
        return {t: 0.0 for t in teams}
    passes = df[df['is_pass'] == 1].dropna(subset=['x', 'y', 'end_x', 'end_y']).copy()
    if passes.empty:
        return {t: 0.0 for t in teams}
    passes['xt'] = passes.apply(xt_lite, axis=1)
    xt_team = passes.groupby('team')['xt'].sum().to_dict()
    return {t: round(float(xt_team.get(t, 0.0)), 2) for t in teams}


# ====== SHOT MAP — PRO ======
def draw_shot_map_pro(shots_df, teams, meta, out_path):
    """Draw shot map for both teams.

    Parameters
    ----------
    shots_df : DataFrame
        Shot events. Must contain at least ``x`` and ``y`` columns.

    Raises
    ------
    ValueError
        If ``shots_df`` is ``None``/empty or lacks required columns.
    """
    required = {"x", "y"}
    if shots_df is None or shots_df.empty:
        raise ValueError("No shot data available to draw shot map")
    if not required.issubset(shots_df.columns):
        missing = ", ".join(sorted(required - set(shots_df.columns)))
        raise ValueError(f"Missing required column(s) for shot map: {missing}")

    team_color = {teams[0]: COLORS['blue'], teams[1]: COLORS['cyan']}

    pitch = Pitch(pitch_type='statsbomb', pitch_color=COLORS['grass'], line_color=COLORS['fog'], linewidth=1)
    fig, ax = pitch.draw()
    fig.set_size_inches(WIDTH_PX / DPI, HEIGHT_PX / DPI)
    add_grass_texture(ax, alpha=0.18)

    for t in teams:
        sub = shots_df.loc[shots_df['team'] == t].dropna(subset=['x', 'y']).copy()
        if sub.empty:
            continue
        on_mask = shot_on_target_mask(sub)
        sizes = scale_sizes(sub['xg'].fillna(0.06) * 100, base=40, k=7.5, min_size=20, max_size=260)
        if (~on_mask).any():
            pitch.scatter(
                sub.loc[~on_mask, 'x'],
                sub.loc[~on_mask, 'y'],
                s=sizes[~on_mask],
                ax=ax,
                zorder=4,
                label=t,
                **shot_marker_kwargs(False, team_color[t]),
            )
        if on_mask.any():
            pitch.scatter(
                sub.loc[on_mask, 'x'],
                sub.loc[on_mask, 'y'],
                s=sizes[on_mask],
                ax=ax,
                zorder=4,
                label=None,
                **shot_marker_kwargs(True, team_color[t]),
            )
        annotate_goals_scatter(ax, sub)

    leg = ax.legend(loc='lower left', frameon=False, fontsize=10)
    if leg is not None and leg.get_title() is not None:
        leg.get_title().set_color(COLORS['fog'])
    ax.set_title(f"Shot Map — {teams[1]} @ {teams[0]}  ({meta.get('date','')})", loc='left', pad=10, fontsize=13)
    annotate_score(ax, teams, meta.get('home_goals', 0), meta.get('away_goals', 0))

    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=20 / DPI, facecolor=COLORS['navy'])
    plt.close(fig)


# ====== XG RACE — PRO ======
def draw_xg_race_pro(shots_df, teams, meta, out_path):
    if shots_df is None or shots_df.empty:
        pitch = Pitch(pitch_color=COLORS['navy'], line_color=COLORS['navy'])
        fig, ax = pitch.draw()
        fig.set_size_inches(WIDTH_PX / DPI, HEIGHT_PX / DPI)
        ax.axis('off')
        ax.text(0.5, 0.5, "Sin tiros registrados", ha='center', va='center', fontsize=14, color=COLORS['fog'])
        fig.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=20 / DPI, facecolor=COLORS['navy'])
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

    pitch = Pitch(pitch_color=COLORS['navy'], line_color=COLORS['navy'])
    fig, ax = pitch.draw()
    fig.set_size_inches(WIDTH_PX / DPI, HEIGHT_PX / DPI)
    ax.cla()
    for t in teams:
        ax.plot(minutes, series[t].values, lw=1.6, alpha=0.95, label=t, color=colors[t])

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

    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=20 / DPI, facecolor=COLORS['navy'])
    plt.close(fig)


# ====== PASSING NETWORK — PRO ======
def draw_pass_network_pro(events_df, teams, meta, kpis, team_focus, out_path):
    """Dibuja la red de pases para un equipo.

    Se requiere la columna ``receiver`` y coordenadas ``x``, ``y``,
    ``end_x`` y ``end_y``. Si los datos son insuficientes se lanza
    :class:`ValueError`.
    """
    required = {"receiver", "x", "y", "end_x", "end_y"}
    if events_df is None or events_df.empty:
        raise ValueError("No event data available to draw passing network")
    if not required.issubset(events_df.columns):
        missing = ", ".join(sorted(required - set(events_df.columns)))
        raise ValueError(f"Missing required column(s) for pass network: {missing}")

    df = events_df.copy()
    df_pass = df[(df.get('is_pass', 0) == 1) & (df.get('event_type') == 'Pass')]
    df_pass = df_pass[df_pass['team'] == team_focus].copy()

    df_pass = df_pass[df_pass['receiver'].notna() & (df_pass['receiver'].astype(str).str.strip() != '')].copy()
    if df_pass.empty:
        raise ValueError("No passing data available after filtering")

    starts = df_pass[['player', 'x', 'y']].rename(columns={'player': 'name'})
    recvs = df_pass[['receiver', 'end_x', 'end_y']].rename(columns={'receiver': 'name', 'end_x': 'x', 'end_y': 'y'})
    touches = pd.concat([starts, recvs], ignore_index=True).dropna(subset=['name', 'x', 'y'])
    locs = touches.groupby('name')[['x', 'y']].mean()
    touch_count = touches.groupby('name').size().rename('touches')

    links = (df_pass.groupby(['player', 'receiver']).size()
             .reset_index(name='count').sort_values('count', ascending=False))
    links = links[links['count'] >= 2]
    prog_mask = (df_pass.get('end_x') - df_pass.get('x')) > 10
    prog_links = (
        df_pass[prog_mask]
        .groupby(['player', 'receiver'])
        .size()
        .reset_index(name='prog_count')
    )
    links = links.merge(prog_links, on=['player', 'receiver'], how='left')
    links['prog_count'] = links.get('prog_count', 0).fillna(0).astype(int)

    pitch = Pitch(pitch_type='statsbomb', pitch_color=COLORS['grass'], line_color=COLORS['fog'], linewidth=1)
    fig, _ = pitch.draw()
    fig.set_size_inches(WIDTH_PX / DPI, HEIGHT_PX / DPI)
    fig.clf()
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[0.33, 0.67], wspace=0.04, figure=fig)
    ax_info = fig.add_subplot(gs[0, 0])
    ax_pitch = fig.add_subplot(gs[0, 1])
    ax_info.set_facecolor(COLORS['navy'])
    ax_info.axis('off')
    pitch.draw(ax=ax_pitch)
    add_grass_texture(ax_pitch, alpha=0.18)

    m_min = int(df_pass['minute'].min()) if 'minute' in df_pass.columns else 0
    m_max = int(df_pass['minute'].max()) if 'minute' in df_pass.columns else 90
    t1, t2 = teams[0], teams[1]
    xg_t1 = float(kpis.get(t1, {}).get('xg', float('nan')))
    xg_t2 = float(kpis.get(t2, {}).get('xg', float('nan')))

    def _layout_text(ax, blocks, x=0.08, top=0.95, bottom=0.05, line_spacing=0.05, block_spacing=0.06):
        total_lines = sum(len(b) for b in blocks)
        needed = total_lines * line_spacing + (len(blocks) - 1) * block_spacing
        available = top - bottom
        if needed > available:
            scale = available / needed
            line_spacing *= scale
            block_spacing *= scale
        y = top
        for i, block in enumerate(blocks):
            for text, style in block:
                ax.annotate(
                    text,
                    xy=(x, y),
                    xycoords='axes fraction',
                    ha='left',
                    va='top',
                    **style,
                )
                y -= line_spacing
            if i < len(blocks) - 1:
                y -= block_spacing

    text_blocks = [
        [
            ("PASSING NETWORK", {"color": "#cfe3ff", "fontsize": 13, "fontweight": "bold"}),
            ("CAPTURE DATA", {"color": "#8aaad6", "fontsize": 9}),
            (f"PLAYER LOCATION: AVERAGE TOUCH POSITION\n{len(df_pass)} PASSES FROM {m_min:02d}' TO {m_max:02d}'",
             {"color": "#cfe3ff", "fontsize": 9, "linespacing": 1.4}),
        ],
        [
            (f"{xg_t1:.1f}", {"color": "#ffffff", "fontsize": 26, "fontweight": "bold"}),
            (f"xG {t1}", {"color": "#8aaad6", "fontsize": 10}),
        ],
        [
            (f"{xg_t2:.1f}", {"color": "#ffffff", "fontsize": 26, "fontweight": "bold"}),
            (f"xG {t2}", {"color": "#8aaad6", "fontsize": 10}),
        ],
        [
            ("● TOU C H E S      ― PASSES", {"color": "#8aaad6", "fontsize": 9}),
        ],
    ]
    _layout_text(ax_info, text_blocks)

    for _, e in links.iterrows():
        a, b, w = e['player'], e['receiver'], int(e['count'])
        prog = int(e.get('prog_count', 0)) > 0
        if a in locs.index and b in locs.index:
            xa, ya = float(locs.loc[a, 'x']), float(locs.loc[a, 'y'])
            xb, yb = float(locs.loc[b, 'x']), float(locs.loc[b, 'y'])
            color = COLORS['goal'] if prog else COLORS['cyan']
            weight = (w + 1 if prog else w) * 0.6
            curved_edge(ax_pitch, xa, ya, xb, yb, weight=weight, color_main=color)

    sizes = scale_sizes(touch_count.reindex(locs.index).fillna(0).astype(int), base=80, k=12, min_size=80, max_size=450)
    pitch.scatter(locs['x'], locs['y'], s=sizes * 1.15, color='#ffffff', alpha=0.08, zorder=3, ax=ax_pitch)
    pitch.scatter(locs['x'], locs['y'], s=sizes, color=COLORS['cyan'], edgecolors=COLORS['fog'],
                  linewidth=0.8, alpha=0.95, zorder=4, ax=ax_pitch)
    for name, row in locs.iterrows():
        label(ax_pitch, row['x'], row['y'], str(name).replace('_', ' '))

    ax_pitch.set_title(f"Passing Network — {team_focus}", loc='left', color="#cfe3ff", fontsize=13, pad=10)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=20 / DPI, facecolor=COLORS['navy'])
    plt.close(fig)


def draw_pressure_map(events_df, teams, meta, out_path):
    df = events_df.copy()
    mask = (
        (df.get('is_def_action', 0) == 1)
        & df['x'].notna()
        & df['y'].notna()
        & (df.get('x') >= 60)
    )
    df = df[mask]
    if df.empty:
        pitch = Pitch(pitch_type='statsbomb', pitch_color=COLORS['grass'], line_color=COLORS['fog'], linewidth=1)
        fig, ax = pitch.draw()
        fig.set_size_inches(WIDTH_PX / DPI, HEIGHT_PX / DPI)
        ax.axis('off')
        ax.text(0.5, 0.5, "Sin acciones defensivas altas", ha='center', va='center', fontsize=14)
        fig.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=20 / DPI, facecolor=COLORS['navy'])
        plt.close(fig)
        return

    team_color = {teams[0]: COLORS['blue'], teams[1]: COLORS['cyan']}
    pitch = Pitch(pitch_type='statsbomb', pitch_color=COLORS['grass'], line_color=COLORS['fog'], linewidth=1)
    fig, ax = pitch.draw()
    fig.set_size_inches(WIDTH_PX / DPI, HEIGHT_PX / DPI)
    add_grass_texture(ax, alpha=0.18)

    for t in teams:
        sub = df[df['team'] == t]
        if sub.empty:
            continue
        pitch.scatter(sub['x'], sub['y'], ax=ax, s=40, color=team_color[t], alpha=0.9, label=t)

    leg = ax.legend(loc='lower left', frameon=False, fontsize=10)
    if leg is not None and leg.get_title() is not None:
        leg.get_title().set_color(COLORS['fog'])
    ax.set_title(f"Pressure Map — {teams[1]} @ {teams[0]}  ({meta.get('date','')})", loc='left', pad=10, fontsize=13)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=20 / DPI, facecolor=COLORS['navy'])
    plt.close(fig)


def process_match(
    events,
    meta,
    output_dir,
    brand_path,
    team_focus=None,
    report_name="river_libertad_report.html",
    img_dir="report/img",
    pbi_dir="powerbi_exports",
):
    img_dir = Path(img_dir)
    if not img_dir.is_absolute():
        img_dir = output_dir / img_dir
    pbi_dir = Path(pbi_dir)
    if not pbi_dir.is_absolute():
        pbi_dir = output_dir / pbi_dir

    report_dir = img_dir.parent
    for p in [report_dir, img_dir, pbi_dir]:
        p.mkdir(parents=True, exist_ok=True)

    teams = [meta.get('home_team', 'Equipo A'), meta.get('away_team', 'Equipo B')]

    required_cols = {
        'match_id', 'team', 'is_shot', 'is_goal', 'is_pass', 'is_def_action',
        'x', 'y', 'end_x', 'end_y', 'xg', 'minute'
    }
    missing = required_cols - set(events.columns)
    if missing:
        missing_str = ', '.join(sorted(missing))
        raise ValueError(f"Missing required column(s): {missing_str}")

    for c in ['is_shot', 'is_goal', 'is_pass', 'is_def_action']:
        events[c] = pd.to_numeric(events[c], errors='coerce').fillna(0).astype(int)
    for c in ['x', 'y', 'end_x', 'end_y', 'xg', 'minute']:
        events[c] = pd.to_numeric(events[c], errors='coerce')

    shots = events[events.get('is_shot', 0) == 1].copy()
    if 'xg' not in shots.columns or shots['xg'].isna().all():
        shots['xg'] = shots.apply(xg_model, axis=1)

    kpis = {}
    for t in teams:
        sub = shots[shots['team'] == t]
        kpis[t] = {
            'shots': int(len(sub)),
            'goals': int(sub.get('is_goal', 0).sum()),
            'xg': float(round(sub.get('xg', 0).sum(), 2)),
        }

    xt_vals = calculate_xt(events, teams)
    for t in teams:
        kpis[t]['xt'] = xt_vals.get(t, 0.0)

    ppda_vals = {t: ppda(events, t) for t in teams}

    shotmap_path = img_dir / "shotmap.png"
    draw_shot_map_pro(shots, teams, meta, shotmap_path)
    print("Guardado:", shotmap_path)

    xgrace_path = img_dir / "xg_race.png"
    draw_xg_race_pro(shots, teams, meta, xgrace_path)
    print("Guardado:", xgrace_path)

    focus = team_focus if team_focus is not None else teams[1]
    passnet_path = img_dir / "pass_network.png"
    draw_pass_network_pro(events, teams, meta, kpis, focus, passnet_path)
    print("Guardado:", passnet_path)

    pressure_path = img_dir / "pressure_map.png"
    draw_pressure_map(events, teams, meta, pressure_path)
    print("Guardado:", pressure_path)

    shots[['match_id', 'team', 'minute', 'x', 'y', 'is_goal', 'xg']].to_csv(
        pbi_dir / "shots.csv", index=False
    )
    pd.DataFrame([
        {"team": t, **kpis[t], "ppda": ppda_vals[t]} for t in teams
    ]).to_csv(pbi_dir / "kpis.csv", index=False)

    report_path = report_dir / report_name
    render_html_report_pro(
        meta,
        teams,
        kpis,
        ppda_vals,
        shotmap_path,
        xgrace_path,
        passnet_path,
        pressure_path,
        brand_path / 'ush_logo_dark.svg',
        report_path,
    )
    print("Listo →", report_path)

    return {
        "shotmap": shotmap_path,
        "xg_race": xgrace_path,
        "pass_network": passnet_path,
        "pressure_map": pressure_path,
        "shots_csv": pbi_dir / "shots.csv",
        "kpis_csv": pbi_dir / "kpis.csv",
        "report": report_path,
    }


def generate_report_for_match(
    events_df,
    matches_df,
    output_dir,
    match_id=None,
    team_focus=None,
    report_name="river_libertad_report.html",
    img_dir="report/img",
    pbi_dir="powerbi_exports",
):
    """Filter matches by ``match_id`` and process them.

    Validates that ``events_df`` contains the columns required by the
    visualizations (``x``, ``y`` and ``receiver``). Returns a list of
    dictionaries with paths to generated files per match.
    """
    ROOT = Path(__file__).resolve().parents[1]
    BRAND = ROOT / "brand"

    output_dir = Path(output_dir)

    plt.style.use("styles/ush_pro.mplstyle")

    needed = {"x", "y", "receiver"}
    missing_cols = needed - set(events_df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"Events DataFrame missing required column(s): {missing}")

    if match_id is not None and "match_id" in matches_df.columns:
        matches_df = matches_df[matches_df["match_id"] == match_id]
        if matches_df.empty:
            raise ValueError(f"No match with id {match_id}")

    results = []
    for idx, row in matches_df.iterrows():
        meta = row.to_dict()
        mid = meta.get("match_id")
        if "match_id" in events_df.columns and mid is not None:
            events = events_df[events_df["match_id"] == mid].copy()
        else:
            events = events_df.copy()
        match_out_dir = output_dir / str(mid if mid is not None else idx)
        paths = process_match(
            events,
            meta,
            match_out_dir,
            BRAND,
            team_focus,
            report_name,
            img_dir,
            pbi_dir,
        )
        results.append({"match_id": mid, **{k: str(v) for k, v in paths.items()}})
    return results


def run_pipeline(events_path, matches_path, out_dir, match_id=None, team_focus=None):
    """Run the pro report pipeline from CSV file paths.

    Parameters
    ----------
    events_path : str or Path
        Ruta al CSV de eventos.
    matches_path : str or Path
        Ruta al CSV de partidos.
    out_dir : str or Path
        Directorio donde se generarán los archivos.
    match_id : optional
        Identificador del partido a procesar. Si es ``None`` se procesan todos.
    team_focus : optional
        Equipo a resaltar en la red de pases.

    Returns
    -------
    list of dict
        Información con rutas a archivos generados por partido.
    """

    events_df = pd.read_csv(Path(events_path))
    matches_df = pd.read_csv(Path(matches_path))
    return generate_report_for_match(
        events_df,
        matches_df,
        Path(out_dir),
        match_id=match_id,
        team_focus=team_focus,
    )


def main():
    """CLI for generating pro reports.

    When called with ``--demo`` synthetic data are generated in ``demo_data/``
    and used as defaults for ``--events``, ``--matches`` and ``--output`` if
    they are not supplied by the user.
    """

    parser = argparse.ArgumentParser(description="Generate Ush Analytics pro report")
    parser.add_argument("--events", help="Path to events CSV (required unless --demo)")
    parser.add_argument("--matches", help="Path to matches CSV (required unless --demo)")
    parser.add_argument("--output", help="Output directory for generated files (required unless --demo)")
    parser.add_argument("--team-focus", help="Team to highlight in passing network")
    parser.add_argument("--match-id", help="Match ID to process (processes all if omitted)")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate synthetic demo data in demo_data/ and use it",
    )
    args = parser.parse_args()

    if args.demo:
        demo_dir = Path(__file__).resolve().parent / "demo_data"
        import generate_synthetic_data as gsd

        gsd.DATA_DIR = demo_dir
        gsd.main()
        if args.events is None:
            args.events = demo_dir / "events.csv"
        if args.matches is None:
            args.matches = demo_dir / "matches.csv"
        if args.output is None:
            args.output = demo_dir / "output"
        print(f"Demo events CSV: {Path(args.events).resolve()}")
        print(f"Demo matches CSV: {Path(args.matches).resolve()}")
        print(f"Output directory: {Path(args.output).resolve()}")

    if not (args.events and args.matches and args.output):
        parser.error("--events, --matches and --output are required unless --demo is used")

    run_pipeline(
        args.events,
        args.matches,
        args.output,
        match_id=args.match_id,
        team_focus=args.team_focus,
    )


if __name__ == "__main__":
    main()

