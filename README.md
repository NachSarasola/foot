# River–Libertad Portfolio Pack (Branded: Ush Analytics)

## Cómo correr (sin Jupyter)
```
pip install -r requirements.txt
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
=======
El informe HTML utiliza un template Jinja ubicado en `templates/ush_report_pro.html`. Si editás ese archivo, los cambios se reflejarán automáticamente al volver a ejecutar el pipeline.

## Jupyter
Abrí `notebooks/01_river_libertad_pospartido.ipynb` y ejecutá todo.

## Branding
Usa `brand/` (logo SVG, avatar, paleta, tema PBI).
