"""
ush_report.py — HTML Report (Ush Pro), standalone

Usage from your run_all.py:
    from ush_report import render_html_report_pro
    render_html_report_pro(meta, teams, kpis, ppda_vals,
                           shotmap_path, xgrace_path, passnet_path,
                           logo_path=BRAND / "ush_logo_dark.svg",
                           out_path=REPORT / "river_libertad_report.html")
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Any
from jinja2 import Environment, FileSystemLoader
import pandas as pd

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"


def render_html_report_pro(meta: Dict[str, Any],
                           teams: Sequence[str],
                           kpis: Dict[str, Dict[str, Any]],
                           ppda_vals: Dict[str, Any],
                           shotmap_path: Path,
                           xgrace_path: Path,
                           passnet_path: Path,
                           logo_path: Path,
                           out_path: Path) -> None:
    """Renderiza el informe HTML Pro (portada + contenido) con branding Ush."""
    required_keys = ["competition", "date", "venue_city"]
    for k in required_keys:
        if k not in meta:
            raise KeyError(f"meta['{k}'] faltante")

    title = f"{teams[1]} @ {teams[0]} — Ida {meta['competition']} ({meta['date']}, {meta['venue_city']})"
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    template = env.get_template("ush_report_pro.html")
    html = template.render(
        title=title,
        logo=str(logo_path),
        comp=meta["competition"], date=meta["date"], venue=meta["venue_city"],
        home=teams[0], away=teams[1],
        home_goals=meta.get("home_goals", ""), away_goals=meta.get("away_goals", ""),
        teams=teams, kpis=kpis, ppda=ppda_vals, team_focus=teams[1],
        shotmap=str(shotmap_path), xgrace=str(xgrace_path), passnet=str(passnet_path),
        year=pd.Timestamp.now().year
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print("Reporte HTML Pro →", out_path)
