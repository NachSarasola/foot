# River–Libertad Portfolio Pack (Branded: Ush Analytics)

## Cómo correr (sin Jupyter)
```
pip install -r requirements.txt
python scripts/run_all_pro.py --events data/events.csv --matches data/matches.csv --output . --team-focus "Equipo B"
```
Salida:
- report/img/shotmap.png, xg_race.png
- report/img/pass_network.png *(se genera al vuelo; no se versiona)*
- report/river_libertad_report.html (2 páginas: tapa + contenido)
- powerbi_exports/shots.csv y kpis.csv

> **Nota**: las funciones de dibujo validan que las columnas
> necesarias estén presentes (`x`, `y`, `receiver`) y lanzan
> `ValueError` si faltan o si no hay datos para graficar.

El archivo `report/img/pass_network.png` se crea al ejecutar el comando anterior y queda excluido del control de versiones.

El informe HTML utiliza un template Jinja ubicado en `templates/ush_report_pro.html`. Si editás ese archivo, los cambios se reflejarán automáticamente al volver a ejecutar el pipeline.

### Ejemplo rápido

Si querés un vistazo rápido con datos sintéticos:

```
python scripts/run_demo.py
```

## Jupyter
Abrí `notebooks/01_river_libertad_pospartido.ipynb` y ejecutá todo.

## Branding
Usa `brand/` (logo SVG, avatar, paleta, tema PBI).

## Power BI

Para mantener el dashboard actualizado:

1. Ejecutá el pipeline anterior para regenerar `powerbi_exports/*.csv`.
2. Abrí `powerbi/dashboard.pbix` en Power BI Desktop.
3. En **Transform Data** → **Data source settings**, apuntá a los CSV exportados (`kpis.csv`, `shots.csv`).
4. Refrescá el modelo.
5. Verificá/creá las medidas DAX principales:
   ```DAX
   xG = SUM(shots[xg])
   xA = SUM(kpis[xa])
   xT_added = SUM(kpis[xt_added])
   PPDA = SUM(kpis[ppda])
   ```
6. Actualizá los visuales:
   - Tarjetas KPI (xG, xA, xT_added, PPDA).
   - Visuales R/Python para Shot Map y Passing Network.
   - Tabla dinámica filtrable por `score_state`.
7. Exportá capturas a `powerbi_exports/`.

