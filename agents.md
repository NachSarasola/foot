# Ush Analytics — agents.md

## Propósito

Este archivo sirve como **guía de trabajo para asistentes de IA (Codex/ChatGPT)** y para cualquier colaborador humano que participe en este proyecto.  
El objetivo es mantener un estándar **profesional, consistente y atractivo visualmente**, alineado con el branding _Ush Analytics_.

## Roles de agentes

- **Visual Lead (VL)**: define look & feel, tipografías, paleta, grids, export settings.
- **Viz Engineer (VE)**: implementa gráficos en Python/mplsoccer/Matplotlib siguiendo el tema.
- **Report Architect (RA)**: mantiene el motor de reportes HTML (Jinja2) y asegura A11y, SEO y formato A4/Responsive.
- **Data Engineer (DE)**: limpia y normaliza `events.csv`, calcula métricas avanzadas (xG, xA, xT, PPDA, EPV, etc.) y genera CSVs para Power BI.
- **QA Analyst (QA)**: valida calidad visual (contraste, tamaños, padding, legibilidad) y consistencia de outputs antes de liberar.

## Estándar visual

- **Fuentes**: Inter (texto), Rajdhani 700 (títulos). Fallback: system-ui, Segoe UI.
- **PNG outputs**: 1600×1000 px, DPI ≥240, `bbox_inches="tight"`.
  - Shot Map: nodos escalados por xG, goles como estrella con etiqueta minuto+jugador.
  - xG Race: curvas lisas con sombreado de ventaja, anotaciones de gol, línea 45’.
  - Passing Network: nodos por toques (clamp min/max), edges curvas ponderadas, labels con fondo sutil.
- **HTML report**: portada + contenido, CSS externo (`report/assets/ush_pro.css`), preparado para imprimir A4 y ver en móvil.

## Estándar de contenido

- **Métricas mínimas**: xG, xA, PPDA, Field Tilt, Shot Maps, Passing Networks, xG Race.
- **Métricas avanzadas** (roadmap): xT, xChain, OBV/EPV, balón parado (xG SP), pressing regains.
- **Power BI**: exportar CSVs normalizados (shots, kpis, xT, regains, etc.) para modelo estrella.
- **Documentación**: cada función con docstring clara, README con guía de instalación/uso y un diccionario de datos.

## Pipeline (DoD: Definition of Done)

1. Cargar datos de `data/`.
2. Calcular KPIs y métricas avanzadas.
3. Generar visuales (PNG) con tema Ush Pro.
4. Exportar CSVs a `powerbi_exports/`.
5. Renderizar informe HTML (`report/*.html`) con portada y branding.
6. QA visual (contraste, padding, coherencia).
7. Commit con mensaje estandarizado (`feat(viz): …`, `fix(data): …`, `style(report): …`).

## QA visual

- Contraste cumple WCAG AA.
- Textos ≥14px, legibles en 13" y en PDF impreso.
- Safe area consistente, nada cortado en HTML/PDF.
- Todas las imágenes con alt text descriptivo.
- Branding (colores, tipografías, logo) aplicado igual en todos los outputs.

## Guía rápida para asistentes

- **No añadir librerías** sin aprobación salvo para QA o linting.
- **Respetar branding** siempre.
- **Mantener helpers** en `ush_style.py` (nunca código duplicado en cada gráfico).
- **Entregar outputs siempre con preview** (PNG + link a HTML).
- **Documentar** cada paso en README o inline docstrings.
