import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from mplsoccer import Pitch

from ush_style import (
    COLORS,
    add_grass_texture,
    scale_sizes,
    text_halo,
    avoid_overlap,
    edge_curved,
    annotate_goals_scatter,
    annotate_goals_on_xg,
    annotate_score,
    shot_marker_kwargs,
    shot_on_target_mask,
    save_fig_pro,
)

WIDTH_PX = 1600
HEIGHT_PX = 1000
DPI = 240


def _save(fig, path: Path, safe: bool = True) -> None:
    """Save figure using project defaults and create thumbnail.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    path : Path or str
        Output path for the main image.
    safe : bool, default ``True``
        If ``True`` apply padding to ensure a safe area using
        :func:`save_fig_pro`. When ``False`` the image is saved without
        additional padding but still generates a thumbnail.
    """
    outfile = Path(path)
    if safe:
        save_fig_pro(fig, outfile, px=(WIDTH_PX, HEIGHT_PX), dpi=DPI)
    else:
        fig.set_size_inches(WIDTH_PX / DPI, HEIGHT_PX / DPI)
        fig.savefig(outfile, dpi=DPI, bbox_inches="tight", facecolor=COLORS["navy"])
        thumb = outfile.with_name(f"{outfile.stem}_thumb.png")
        fig.set_size_inches(800 / DPI, 500 / DPI)
        fig.savefig(thumb, dpi=DPI, bbox_inches="tight", facecolor=COLORS["navy"])


def draw_shot_map(shots_df: pd.DataFrame, teams: list[str], meta: dict, out_path: Path,
                   *, show_title: bool = True, safe: bool = True) -> None:
    """Draw a simple shot map for two teams.

    Parameters
    ----------
    shots_df : DataFrame
        Shot events. Must contain at least ``x`` and ``y`` columns.
    teams : list[str]
        Names of the home and away teams.
    meta : dict
        Metadata including optional ``home_goals``, ``away_goals`` and
        ``date`` entries.
    out_path : Path or str
        Destination file for the generated image.
    show_title : bool, default ``True``
        If ``False`` the title is omitted.
    safe : bool, default ``True``
        Forwarded to :func:`_save` to toggle safe area padding.
    """
    plt.style.use("styles/ush_pro.mplstyle")
    required = {"x", "y"}
    if shots_df is None or shots_df.empty:
        raise ValueError("No shot data available to draw shot map")
    if not required.issubset(shots_df.columns):
        missing = ", ".join(sorted(required - set(shots_df.columns)))
        raise ValueError(f"Missing required column(s) for shot map: {missing}")

    team_color = {teams[0]: COLORS['blue'], teams[1]: COLORS['cyan']}

    pitch = Pitch(pitch_type='statsbomb', pitch_color=COLORS['grass'],
                  line_color=COLORS['fog'], linewidth=1)
    fig, ax = pitch.draw()
    fig.set_size_inches(WIDTH_PX / DPI, HEIGHT_PX / DPI)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.04)
    add_grass_texture(ax, alpha=0.18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()

    for t in teams:
        sub = shots_df.loc[shots_df['team'] == t].dropna(subset=['x', 'y']).copy()
        if sub.empty:
            continue
        on_mask = shot_on_target_mask(sub)
        sizes = scale_sizes(sub.get('xg', 0.06).fillna(0.06) * 100, base=40,
                            k=7.5, min_size=40, max_size=520)
        if (~on_mask).any():
            pitch.scatter(
                sub.loc[~on_mask, 'x'],
                sub.loc[~on_mask, 'y'],
                s=sizes[~on_mask],
                ax=ax,
                zorder=2,
                label=t,
                **shot_marker_kwargs(False, team_color[t]),
            )
        if on_mask.any():
            pitch.scatter(
                sub.loc[on_mask, 'x'],
                sub.loc[on_mask, 'y'],
                s=sizes[on_mask],
                ax=ax,
                zorder=2,
                label=None,
                **shot_marker_kwargs(True, team_color[t]),
            )
        annotate_goals_scatter(ax, sub)

    leg = ax.legend(
        loc='upper left',
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        fontsize=10,
        handlelength=1,
        handletextpad=0.4,
        borderpad=0.2,
    )
    if leg is not None and leg.get_title() is not None:
        leg.get_title().set_color(COLORS['fog'])
    if show_title:
        ax.set_title(
            f"Shot Map — {teams[1]} @ {teams[0]}  ({meta.get('date','')})",
            loc='left',
            pad=10,
            fontsize=13,
        )
    annotate_score(ax, teams, meta.get('home_goals', 0), meta.get('away_goals', 0))

    _save(fig, out_path, safe=safe)
    plt.close(fig)


def draw_xg_race(shots_df: pd.DataFrame, teams: list[str], meta: dict, out_path: Path,
                 *, show_title: bool = True, safe: bool = True) -> None:
    """Draw cumulative expected goals over time for two teams."""
    plt.style.use("styles/ush_pro.mplstyle")
    if shots_df is None or shots_df.empty:
        pitch = Pitch(pitch_color=COLORS['navy'], line_color=COLORS['navy'])
        fig, ax = pitch.draw()
        fig.set_size_inches(WIDTH_PX / DPI, HEIGHT_PX / DPI)
        fig.subplots_adjust(left=0.06, top=0.9)
        ax.axis('off')
        text_halo(ax, "Sin tiros registrados", x=0.5, y=0.5,
                  ha='center', va='center', fontsize=14, color=COLORS['ink'])
        _save(fig, out_path, safe=safe)
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
    fig.subplots_adjust(left=0.06, top=0.9)
    ax.cla()
    for t in teams:
        ax.plot(
            minutes,
            series[t].values,
            lw=1.6,
            alpha=0.95,
            label=t,
            color=colors[t],
            solid_capstyle="round",
        )

    lead = series[teams[1]] - series[teams[0]]
    ax.fill_between(minutes, series[teams[0]], series[teams[1]],
                    where=(lead >= 0), interpolate=True, color=COLORS['cyan'], alpha=0.08, zorder=0)
    ax.fill_between(minutes, series[teams[0]], series[teams[1]],
                    where=(lead < 0), interpolate=True, color=COLORS['blue'], alpha=0.08, zorder=0)

    annotate_goals_on_xg(ax, df, teams[0], colors[teams[0]])
    annotate_goals_on_xg(ax, df, teams[1], colors[teams[1]])
    ax.axvline(45, color=COLORS['fog'], alpha=0.12, lw=1)
    ax.text(
        45,
        ax.get_ylim()[1] * 0.05,
        "HT",
        color=COLORS['fog'],
        ha='center',
        va='bottom',
        fontsize=9,
    )

    ax.set_xlim(0, minutes[-1])
    ax.set_xlabel('Minuto')
    ax.set_ylabel('xG acumulado')
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.legend(
        loc='lower left',
        bbox_to_anchor=(0, 1.02),
        ncol=2,
        frameon=False,
        borderaxespad=0,
        handlelength=1.2,
        columnspacing=1.2,
    )
    if show_title:
        ax.set_title(
            f"xG Race — {teams[1]} @ {teams[0]}  ({meta.get('date','')})",
            loc='left',
            pad=10,
            fontsize=13,
        )

    _save(fig, out_path, safe=safe)
    plt.close(fig)


def draw_passing_network(events_df: pd.DataFrame, teams: list[str], meta: dict,
                          kpis: dict, team_focus: str, out_path: Path,
                          *, show_title: bool = True, safe: bool = True) -> None:
    """Draw a passing network for ``team_focus``.

    Parameters
    ----------
    events_df : DataFrame
        Event data containing pass information.
    teams : list[str]
        Names of the two teams.
    meta : dict
        Match metadata (date, score, etc.).
    kpis : dict
        Dictionary with per-team metrics such as xG.
    team_focus : str
        Team to highlight.
    out_path : Path or str
        Destination image path.
    show_title : bool, default ``True``
        Whether to draw the title.
    safe : bool, default ``True``
        Passed to :func:`_save`.
    """
    required = {"receiver", "x", "y", "end_x", "end_y"}
    plt.style.use("styles/ush_pro.mplstyle")
    if events_df is None or events_df.empty:
        raise ValueError("No event data available to draw passing network")
    if not required.issubset(events_df.columns):
        missing = ", ".join(sorted(required - set(events_df.columns)))
        raise ValueError(f"Missing required column(s) for pass network: {missing}")

    df = events_df.copy()
    df_pass = df[
        (df.get('is_pass', 0) == 1)
        & (df.get('event_type', '').astype(str).str.lower() == 'pass')
    ]
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
            ("● TOUCHES — PASSES", {"color": "#8aaad6", "fontsize": 9}),
        ],
    ]
    _layout_text(ax_info, text_blocks)

    max_w = links['count'].max() if not links.empty else 1

    for _, e in links.iterrows():
        a, b, w = e['player'], e['receiver'], int(e['count'])
        prog = int(e.get('prog_count', 0)) > 0
        if a in locs.index and b in locs.index:
            xa, ya = float(locs.loc[a, 'x']), float(locs.loc[a, 'y'])
            xb, yb = float(locs.loc[b, 'x']), float(locs.loc[b, 'y'])
            color = COLORS['goal'] if prog else COLORS['cyan']
            weight = max(2.0, (w + 1 if prog else w) * 0.6)
            alpha = 0.3 + 0.7 * (w / max_w)
            edge_curved(
                ax_pitch,
                (xa, ya),
                (xb, yb),
                weight=weight,
                color=color,
                alpha=alpha,
                shadow=True,
            )

    sizes = scale_sizes(
        touch_count.reindex(locs.index).fillna(0).astype(int),
        base=80,
        k=12,
        min_size=80,
        max_size=450,
    )
    pitch.scatter(
        locs['x'],
        locs['y'],
        s=sizes * 1.3,
        color=COLORS['paper'],
        alpha=0.12,
        zorder=3,
        ax=ax_pitch,
    )
    pitch.scatter(
        locs['x'],
        locs['y'],
        s=sizes,
        color=COLORS['cyan'],
        edgecolors=COLORS['fog'],
        linewidth=0.8,
        alpha=0.95,
        zorder=4,
        ax=ax_pitch,
    )
    texts = []
    for name, row in locs.iterrows():
        txt = text_halo(
            ax_pitch,
            str(name).replace('_', ' '),
            x=row['x'],
            y=row['y'] - 3,
            ha='center',
            va='top',
            color=COLORS['fog'],
            fontsize=9,
            bbox={
                'boxstyle': 'round,pad=0.2',
                'facecolor': COLORS['navy'],
                'alpha': 0.8,
                'edgecolor': 'none',
            },
        )
        texts.append(txt)
    avoid_overlap(texts, padding=6)

    if show_title:
        ax_pitch.set_title(
            f"Passing Network — {team_focus} vs {teams[0] if team_focus == teams[1] else teams[1]}",
            loc='left',
            pad=10,
            fontsize=13,
        )

    _save(fig, out_path, safe=safe)
    plt.close(fig)


def main():
    """CLI for generating the three standard visualisations."""
    parser = argparse.ArgumentParser(description="Generate basic Ush Analytics visuals")
    parser.add_argument("--shots", required=True, help="Path to shots CSV")
    parser.add_argument("--events", required=True, help="Path to events CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--team-focus", required=True, help="Team to highlight in passing network")
    parser.add_argument("--date", default="", help="Match date string")
    parser.add_argument("--no-title", action="store_true", help="Disable chart titles")
    parser.add_argument("--unsafe", action="store_true", help="Disable safe area padding")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    shots = pd.read_csv(args.shots)
    events = pd.read_csv(args.events)

    teams = list(pd.Series(shots['team']).dropna().unique())
    if len(teams) != 2:
        raise ValueError("Shots CSV must contain exactly two teams")

    meta = {
        'home_goals': int(shots[(shots['team'] == teams[0]) & (shots.get('is_goal', 0) == 1)].shape[0]),
        'away_goals': int(shots[(shots['team'] == teams[1]) & (shots.get('is_goal', 0) == 1)].shape[0]),
        'date': args.date,
    }
    kpis = {t: {'xg': float(shots[shots['team'] == t].get('xg', 0).sum())} for t in teams}

    draw_shot_map(shots, teams, meta, out_dir / 'shot_map.png',
                  show_title=not args.no_title, safe=not args.unsafe)
    draw_xg_race(shots, teams, meta, out_dir / 'xg_race.png',
                 show_title=not args.no_title, safe=not args.unsafe)
    draw_passing_network(events, teams, meta, kpis, args.team_focus, out_dir / 'passing_network.png',
                         show_title=not args.no_title, safe=not args.unsafe)


if __name__ == "__main__":
    main()
