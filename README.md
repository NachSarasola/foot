# River–Libertad Portfolio Pack (Branded: Ush Analytics)

## Cómo correr (sin Jupyter)
```
py -m pip install --user pandas numpy matplotlib mplsoccer jinja2
py scripts/run_all.py
```
Salida:
- report/img/shotmap.png, xg_race.png, pass_network.png
- report/river_libertad_report.html (2 páginas: tapa + contenido)
- powerbi_exports/shots.csv y kpis.csv

## Pipeline Pro (CLI)
```
py scripts/run_all_pro.py --events data/events.csv --matches data/matches.csv --output . --team-focus "Equipo B"
```
Genera la misma salida que el script anterior pero permite pasar rutas de entrada, directorio de salida y el equipo enfocado.

## Jupyter
Abrí `notebooks/01_river_libertad_pospartido.ipynb` y ejecutá todo.

## Branding
Usa `brand/` (logo SVG, avatar, paleta, tema PBI).
