# run_all.py — Ush Analytics (fixed)
# Ejecutá este archivo desde VS Code con el botón ▶️ (Run Python File)
# Se instala lo necesario automáticamente si falta.

# --- opcional: auto-instalar librerías si no están ---
import sys, subprocess, importlib
for pkg in ["pandas","numpy","matplotlib","mplsoccer","jinja2"]:
    try:
        importlib.import_module(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", pkg])

from pathlib import Path
import pandas as pd, numpy as np, math
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from jinja2 import Template

ROOT = Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; REPORT=ROOT/'report'; IMG=REPORT/'img'; PBI=ROOT/'powerbi_exports'; BRAND=ROOT/'brand'
for p in [REPORT, IMG, PBI]: p.mkdir(parents=True, exist_ok=True)

events=pd.read_csv(DATA/'events.csv')
matches=pd.read_csv(DATA/'matches.csv')
meta=matches.iloc[0].to_dict()
teams=[meta['home_team'], meta['away_team']]

for c in ['is_shot','is_goal','is_pass','is_def_action']:
    if c in events.columns: events[c]=pd.to_numeric(events[c], errors='coerce').fillna(0).astype(int)
for c in ['x','y','xg']:
    if c in events.columns: events[c]=pd.to_numeric(events[c], errors='coerce')

def xg_lite(row, xmax=120.0, ymax=80.0):
    dx=max(0.0, xmax-float(row.get('x',0))); dy=float(row.get('y',0))-ymax/2.0
    dist=(dx*dx+dy*dy)**0.5; ang=np.arctan2(abs(dy), max(1e-6, dx))
    val=1/(1+np.exp((dist-18)/4))*(0.6+0.4*(1-ang/1.57))
    return max(0.02, min(0.8, float(val)))

shots=events[events['is_shot']==1].copy()
if 'xg' not in shots.columns or shots['xg'].isna().all():
    shots['xg']=shots.apply(xg_lite, axis=1)

# KPIs
kpis={}
for t in teams:
    sub=shots[shots['team']==t]
    kpis[t]={'shots':int(len(sub)),'goals':int(sub['is_goal'].sum()),'xg':float(round(sub['xg'].sum(),2))}

# PPDA básico
def ppda(df, team):
    if not set(['is_pass','is_def_action','y']).issubset(df.columns): return None
    ymax=max(100.0, float(df['y'].max()) or 100.0); yn=df['y']/ymax*100.0
    opp=((df['team']!=team)&(df['is_pass']==1)&(yn>=40)).sum()
    our=((df['team']==team)&(df['is_def_action']==1)&(yn>=40)).sum()
    return round(float(opp)/float(our),2) if our else None
ppda_vals={t:ppda(events,t) for t in teams}

# Shot map
pitch=Pitch(pitch_type='statsbomb', line_zorder=2); fig,ax=pitch.draw(figsize=(8,6))
for i,t in enumerate(teams):
    sub=shots[shots['team']==t]; s=(sub['xg'].fillna(0.06)*900).clip(lower=30)
    pitch.scatter(sub['x'],sub['y'],s=s,marker=['o','s'][i%2],alpha=0.7,label=t,ax=ax)
ax.legend(loc='lower left'); ax.set_title(f"Shot Map — {teams[0]} vs {teams[1]} ({meta['date']})")
out_shot=IMG/'shotmap.png'; plt.savefig(out_shot,dpi=200,bbox_inches='tight'); plt.close(fig)

# xG race
shots['minute']=pd.to_numeric(shots['minute'], errors='coerce').fillna(0).astype(int)
shots['xg_val']=shots['xg'].fillna(0.06); mins=range(0, max(int(shots['minute'].max()),95)+1)
fig,ax=plt.subplots(figsize=(8,4.5))
for t in teams:
    tmp=shots[shots['team']==t].groupby('minute')['xg_val'].sum().reindex(mins, fill_value=0).cumsum()
    ax.plot(list(mins), tmp.values, label=t)
ax.set_xlabel('Minuto'); ax.set_ylabel('xG acumulado'); ax.legend()
out_xg=IMG/'xg_race.png'; plt.savefig(out_xg,dpi=200,bbox_inches='tight'); plt.close(fig)

# Pass network (fallback sin receiver) — FIXED multiline if
passes=events[(events['is_pass']==1) & (events['event_type']=='Pass')].dropna(subset=['player'])
team_focus=teams[1]; df=passes[passes['team']==team_focus].copy()
fig,ax=plt.subplots(figsize=(8,6)); pitch=Pitch(pitch_type='statsbomb', line_zorder=2); pitch.draw(ax=ax)
out_net=IMG/'pass_network.png'
if len(df)>0:
    locs=df.groupby('player')[['x','y']].mean()
    counts=df['player'].value_counts()
    players=list(counts.index)
    rows=[]
    for i, a in enumerate(players):
        for b in players[i+1:]:
            w = int((counts[a] + counts[b]) / 2)
            if w >= 3:
                rows.append((a, b, w))
    for p,row in locs.iterrows():
        pitch.scatter([row['x']],[row['y']], s=300, alpha=0.85, ax=ax)
        ax.text(row['x'],row['y']-3,p,ha='center',va='top',fontsize=9)
    for a,b,w in rows:
        if a in locs.index and b in locs.index:
            xa,ya=locs.loc[a,['x','y']]; xb,yb=locs.loc[b,['x','y']]
            ax.plot([xa,xb],[ya,yb], linewidth=1+0.2*w, alpha=0.4)
    ax.set_title(f"Red de pases — {team_focus}")
plt.savefig(out_net,dpi=200,bbox_inches='tight'); plt.close(fig)

# Exports PBI
shots[['match_id','team','minute','x','y','is_goal','xg']].to_csv(PBI/'shots.csv', index=False)
pd.DataFrame([{'team':t, **kpis[t], 'ppda':ppda_vals[t]} for t in teams]).to_csv(PBI/'kpis.csv', index=False)

# Informe HTML con portada (branding Ush Analytics)
HTML = r"""<!doctype html><html lang='es'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>{{ title }}</title>
<style>
:root{--navy:#0A2540;--cyan:#00C2FF;--blue:#5B86E5;--paper:#FFFFFF;--fog:#E6EEF6;--ink:#0A2540;--goal:#FF3366}
*{box-sizing:border-box} body{font-family:Inter,system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#F6FAFF;color:var(--ink)}
.wrap{max-width:980px;margin:0 auto;padding:16px}
.card{background:var(--paper);border-radius:16px;box-shadow:0 6px 24px rgba(10,37,64,.08);padding:20px;margin:16px 0}
h1{margin:0;font-size:32px} h2{margin-top:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.kpi{background:#EEF4FF;border-radius:12px;padding:12px}
.kpi b{font-size:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
img{max-width:100%;border-radius:12px}
.badge{display:inline-flex;align-items:center;gap:12px;background:var(--navy);color:#fff;border-radius:999px;padding:6px 12px}
.badge img{height:28px}
footer{font-size:12px;color:#567;text-align:center;padding:24px}
.cover{background:linear-gradient(135deg,var(--navy),#081a30);color:#fff;padding:48px 0}
.cover .title{display:flex;align-items:center;gap:16px}
.cover h1{font-size:40px}
.tag{display:inline-block;background:var(--cyan);color:#001018;padding:6px 10px;border-radius:999px;font-weight:600;margin-left:8px}
.meta{margin-top:12px;color:#cfe3ff}
hr.sep{border:none;border-top:2px solid var(--fog);margin:8px 0}
@media print{@page{size:A4;margin:12mm} .pagebreak{page-break-before:always} a[href]:after{content:''}}
</style></head>
<body>
  <section class="cover">
    <div class="wrap">
      <div class="title">
        <img src="{{ logo }}" alt="Ush Analytics" style="height:72px">
        <div>
          <h1>Informe pospartido <span class="tag">Ush Analytics</span></h1>
          <div>{{ comp }} — {{ date }} — {{ venue }}</div>
        </div>
      </div>
      <div class="meta">
        <div><b>Partido:</b> {{ away }} @ {{ home }} — Ida</div>
        <div><b>Marcador:</b> {{ away_goals }}–{{ home_goals }}</div>
      </div>
    </div>
  </section>

  <div class="wrap pagebreak">
    <section class="card">
      <h2>KPIs</h2>
      <div class="kpis">
        {% for t in teams %}
        <div class="kpi">
          <div><b>{{ t }}</b></div>
          <div>Tiros: {{ kpis[t].shots }}</div>
          <div>Goles: {{ kpis[t].goals }}</div>
          <div>xG: {{ "%.2f"|format(kpis[t].xg) }}</div>
          <div>PPDA: {{ ppda[t] if ppda[t] is not none else "—" }}</div>
        </div>
        {% endfor %}
      </div>
    </section>

    <section class="card">
      <h2>Visuales</h2>
      <div class="grid">
        <div><h3>Shot Map</h3><img src="{{ shotmap }}" alt="Shot Map"></div>
        <div><h3>xG acumulado</h3><img src="{{ xgrace }}" alt="xG acumulado"></div>
        <div style="grid-column:1/-1">
          <h3>Red de pases — {{ team_focus }}</h3>
          <img src="{{ passnet }}" alt="Red de pases">
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Notas rápidas</h2>
      <ul>
        <li><b>Momentum:</b> River mejoró su xG en el ST y marcó en momentos clave <span style="color:var(--goal)">({{ goals_away }})</span>.</li>
        <li><b>Construcción:</b> Libertad priorizó mitad de cancha; pressing selectivo de River (PPDA más bajo).</li>
        <li><b>Circulación:</b> Foco en carril derecho para {{ team_focus }} con conexiones fuertes interior–extremo.</li>
      </ul>
    </section>
  </div>

  <footer>© Ush Analytics — Generado automáticamente.</footer>
</body></html>
"""

title=f"{teams[1]} vs {teams[0]} — Ida {meta['competition']} ({meta['date']}, {meta['venue_city']})"
html=Template(HTML).render(
    title=title, logo=str(BRAND/'ush_logo_dark.svg'), comp=meta['competition'], date=meta['date'],
    venue=meta['venue_city'], away=teams[1], home=teams[0], away_goals=meta['away_goals'], home_goals=meta['home_goals'],
    teams=teams, kpis=kpis, ppda=ppda_vals, shotmap=str(out_shot), xgrace=str(out_xg), passnet=str(out_net),
    team_focus=teams[1], goals_away=meta['away_goals']
)
(REPORT/'river_libertad_report.html').write_text(html, encoding='utf-8')
print("Listo →", REPORT/'river_libertad_report.html')
