"""CSS lifted from sky_detection_console_final.html, adapted for Streamlit's DOM.
Dark-only premium theme: black/blue/grey palette (paired with .streamlit/config.toml
forcing base="dark" so native widgets match without per-widget overrides)."""

CSS = """
<style>
:root{
  --bg:#0A0D13;
  --bg-elevated:#0D111A;
  --panel:#10141C;
  --panel-alt:#0C0F16;
  --panel-hover:#161B26;
  --border:#1D2330;
  --border-strong:#2B3342;
  --text-hi:#EDF1F7;
  --text-mid:#98A2B3;
  --text-low:#5C6675;
  --accent:#4C8DFF;
  --accent-bright:#7CAAFF;
  --accent-bg:rgba(76,141,255,0.12);
  --accent-border:rgba(76,141,255,0.45);
  --status-safe:#34D399;
  --status-safe-bg:rgba(52,211,153,0.12);
  --status-caution:#FBBF24;
  --status-caution-bg:rgba(251,191,36,0.12);
  --status-threat:#F87171;
  --status-threat-bg:rgba(248,113,113,0.12);
  --shadow:0 8px 24px rgba(0,0,0,0.45);
  --shadow-sm:0 2px 8px rgba(0,0,0,0.35);
  --radius:8px;
  --mono:'JetBrains Mono','Consolas',monospace;
}

.stApp{
  background:
    radial-gradient(ellipse 1200px 600px at 15% -10%, rgba(76,141,255,0.07), transparent 60%),
    radial-gradient(ellipse 900px 500px at 100% 0%, rgba(124,170,255,0.05), transparent 55%),
    var(--bg);
}
[data-testid="stSidebar"]{ display:none; }
[data-testid="stHeader"]{ background:transparent; }
.stApp, .stApp *{ scrollbar-color: var(--border-strong) var(--bg-elevated); }

.sdt-header{
  display:flex; align-items:center; justify-content:space-between;
  background:linear-gradient(180deg, var(--panel-hover), var(--panel));
  border:1px solid var(--border); border-radius:var(--radius);
  padding:13px 22px; margin-bottom:16px; box-shadow:var(--shadow-sm);
  position:relative; overflow:hidden;
}
.sdt-header::before{
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg, transparent, var(--accent), transparent);
  opacity:0.6;
}
.sdt-brand{ display:flex; align-items:center; gap:11px; }
.sdt-logo{ width:32px; height:32px; border-radius:var(--radius);
  background:linear-gradient(135deg, var(--accent), #2F5FC4);
  display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700; font-size:11.5px;
  box-shadow:0 0 0 1px rgba(76,141,255,0.3), 0 4px 12px rgba(76,141,255,0.25); letter-spacing:0.3px; }
.sdt-brand-name{ font-size:14.5px; font-weight:600; color:var(--text-hi); letter-spacing:0.2px; }
.sdt-brand-sub{ font-size:10.5px; color:var(--text-low); font-family:var(--mono); letter-spacing:0.3px; }
.sdt-header-right{ display:flex; align-items:center; gap:14px; }
.sdt-classes{ display:flex; align-items:center; gap:7px; font-size:11px; color:var(--text-mid); }
.class-pill{ font-weight:600; font-size:10.5px; padding:3px 9px; border-radius:5px;
  border:1px solid var(--border-strong); color:var(--text-mid); font-family:var(--mono); background:var(--bg-elevated); }
.status-pill{ display:flex; align-items:center; gap:6px; font-size:10.5px; color:var(--status-safe);
  border:1px solid rgba(52,211,153,0.35); padding:5px 12px; border-radius:var(--radius);
  font-family:var(--mono); background:var(--status-safe-bg); white-space:nowrap; }
.status-dot{ width:6px; height:6px; border-radius:50%; background:var(--status-safe); display:inline-block;
  box-shadow:0 0 8px var(--status-safe); }

.sdt-panel{ background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
  padding:17px; box-shadow:var(--shadow-sm); height:100%; }
.panel-title{ font-size:12.5px; font-weight:700; color:var(--text-hi); letter-spacing:0.2px; }
.panel-sub{ font-size:11px; color:var(--text-low); margin-top:3px; line-height:1.5; margin-bottom:12px; }
.section-label{ font-size:10px; font-weight:700; letter-spacing:0.6px; color:var(--text-low);
  text-transform:uppercase; margin:16px 0 8px; font-family:var(--mono); }

.source-tabs{ background:var(--panel-alt); border:1px solid var(--border); border-radius:var(--radius);
  padding:8px 10px; margin-bottom:4px; }
.source-tabs [data-testid="stRadio"] label p{
  font-size:11px !important; font-weight:700 !important; letter-spacing:0.3px;
}

.mode-card{ border:1.5px solid var(--border); border-radius:var(--radius); padding:11px 12px;
  margin-top:8px; background:var(--panel-alt); transition:border-color 0.15s; }
.mode-card.selected{ border-color:var(--accent-border); background:var(--accent-bg);
  box-shadow:0 0 0 1px rgba(76,141,255,0.15), 0 4px 14px rgba(76,141,255,0.12); }
.mode-head{ display:flex; justify-content:space-between; align-items:center; }
.mode-name{ font-size:12px; font-weight:700; color:var(--text-hi); }
.mode-badge{ font-size:9px; font-weight:700; padding:2px 8px; border-radius:4px; border:1px solid var(--accent-border);
  color:var(--accent-bright); font-family:var(--mono); background:rgba(76,141,255,0.08); }
.mode-desc{ font-size:10.5px; color:var(--text-mid); margin-top:4px; line-height:1.5; }

.queue-item{ display:flex; gap:10px; align-items:center; padding:9px; border-radius:var(--radius);
  border:1px solid var(--border); background:var(--panel-alt); margin-top:8px; }
.queue-thumb{ width:30px; height:30px; border-radius:5px; background:var(--border-strong); flex-shrink:0; }
.queue-name{ font-size:11px; font-weight:600; color:var(--text-hi); }
.queue-meta{ font-size:9.5px; color:var(--text-low); font-family:var(--mono); }
.queue-status{ margin-left:auto; font-family:var(--mono); font-size:9px; color:var(--status-safe); }

.ws-eyebrow{ font-size:10px; font-weight:700; color:var(--accent-bright); letter-spacing:0.6px; font-family:var(--mono); }
.ws-title{ font-size:14.5px; font-weight:600; margin-top:2px; color:var(--text-hi); }

.hud-chip{ font-family:var(--mono); font-size:10.5px; background:rgba(0,0,0,0.55); color:#DCE4EC;
  padding:4px 9px; border-radius:4px; display:inline-block; }

.metric-box{ margin-top:14px; border:1px solid var(--border); border-radius:var(--radius); padding:11px 12px;
  background:var(--panel-alt); }
.metric-title{ font-size:10px; font-weight:700; color:var(--text-mid); letter-spacing:0.4px; font-family:var(--mono); }
.latency-row{ display:flex; justify-content:space-between; margin-top:10px; font-family:var(--mono); font-size:10.5px; }
.latency-item{ display:flex; flex-direction:column; gap:2px; }
.latency-label{ color:var(--text-low); font-size:9px; text-transform:uppercase; letter-spacing:0.3px; }
.latency-val{ font-weight:700; color:var(--accent-bright); }

.vlm-panel{ margin-top:14px; background:var(--panel-alt); border:1px solid var(--accent-border);
  border-radius:var(--radius); padding:12px 13px; box-shadow:0 4px 14px rgba(76,141,255,0.08); }
.vlm-panel-head{ display:flex; align-items:center; justify-content:space-between; }
.vlm-panel-title{ font-size:10.5px; font-weight:700; letter-spacing:0.5px; color:var(--accent-bright); font-family:var(--mono); }
.vlm-panel-lat{ font-family:var(--mono); font-size:10.5px; color:var(--accent-bright); }
.vlm-panel-sub{ font-size:9.5px; color:var(--text-low); margin-top:5px; font-family:var(--mono); }
.vlm-panel-cls{ font-size:14px; font-weight:700; margin-top:9px; color:var(--text-hi); }
.vlm-panel-conf{ font-family:var(--mono); font-size:11px; font-weight:400; color:var(--accent-bright); }
.vlm-panel-reason{ font-size:11px; color:var(--text-mid); margin-top:7px; line-height:1.5; }

.ws-footer{ font-size:11px; color:var(--text-low); font-family:var(--mono); margin-top:10px; padding-top:10px;
  border-top:1px solid var(--border); }

.final-badge-threat{ color:var(--status-threat); border:1px solid rgba(248,113,113,0.4);
  background:var(--status-threat-bg); padding:3px 10px; border-radius:4px; font-family:var(--mono); font-size:10.5px; font-weight:700; }
.final-badge-safe{ color:var(--status-safe); border:1px solid rgba(52,211,153,0.4);
  background:var(--status-safe-bg); padding:3px 10px; border-radius:4px; font-family:var(--mono); font-size:10.5px; font-weight:700; }
.final-badge-caution{ color:var(--status-caution); border:1px solid rgba(251,191,36,0.4);
  background:var(--status-caution-bg); padding:3px 10px; border-radius:4px; font-family:var(--mono); font-size:10.5px; font-weight:700; }

.custom-model-item{ display:flex; align-items:center; justify-content:space-between;
  padding:9px 11px; border:1px solid var(--border); border-radius:6px; background:var(--panel-alt); }
.cmi-name{ font-family:var(--mono); font-size:11px; color:var(--text-hi); font-weight:600; }
.cmi-meta{ font-family:var(--mono); font-size:9px; color:var(--text-low); margin-top:2px; }
.cmi-badge{ font-family:var(--mono); font-size:9px; padding:2px 8px; border-radius:4px; color:var(--status-safe);
  border:1px solid rgba(52,211,153,0.4); background:var(--status-safe-bg); white-space:nowrap; }

/* Native Streamlit widgets already pick up the dark palette from
   .streamlit/config.toml; these overrides just tune specific pieces of
   widget chrome to match the panel surfaces exactly. */
[data-testid="stRadio"] *,
[data-testid="stSelectbox"] *,
[data-testid="stSlider"] *,
[data-testid="stWidgetLabel"] p,
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stCaptionContainer"] *{
  color:var(--text-hi) !important;
}
[data-testid="stRadio"] label p{
  font-family:var(--mono) !important; font-size:11.5px !important; font-weight:600 !important;
}
[data-testid="stCaptionContainer"] *{
  color:var(--text-low) !important;
}
[data-testid="stRadio"] [role="radiogroup"]{ gap:4px; }

[data-testid="stFileUploaderDropzone"]{
  background:var(--panel-alt) !important; border:1.5px dashed var(--border-strong) !important;
  border-radius:var(--radius) !important;
}
[data-testid="stFileUploaderDropzone"] button{
  background:var(--panel-hover) !important; color:var(--text-hi) !important;
  border:1px solid var(--border-strong) !important; border-radius:6px !important;
}
[data-testid="stSelectbox"] > div > div{
  background:var(--panel-alt) !important; border-color:var(--border-strong) !important;
  border-radius:6px !important;
}
div[data-baseweb="popover"] *{
  color:var(--text-hi) !important;
}
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"]{
  background:var(--panel-hover) !important; border:1px solid var(--border-strong) !important;
}
div[data-baseweb="popover"] li:hover{
  background:var(--accent-bg) !important;
}

/* Buttons: give the primary action a subtle glow befitting a "premium" feel */
button[kind="primary"]{
  background:linear-gradient(135deg, var(--accent), #2F5FC4) !important;
  border:none !important; box-shadow:0 4px 14px rgba(76,141,255,0.3) !important;
  border-radius:7px !important; font-weight:600 !important;
}
button[kind="primary"]:hover{
  box-shadow:0 6px 20px rgba(76,141,255,0.45) !important;
  transform:translateY(-1px);
}
button[kind="secondary"]{
  background:var(--panel-alt) !important; border:1px solid var(--border-strong) !important;
  border-radius:7px !important; color:var(--text-hi) !important;
}

[data-testid="stCodeBlock"]{
  background:var(--bg-elevated) !important; border:1px solid var(--border) !important;
  border-radius:var(--radius) !important;
}
[data-testid="stDataFrame"]{
  border:1px solid var(--border) !important; border-radius:var(--radius) !important; overflow:hidden;
}
[data-testid="stTabs"] [data-baseweb="tab-list"]{
  border-bottom:1px solid var(--border) !important;
}
[data-testid="stTabs"] [aria-selected="true"]{
  color:var(--accent-bright) !important;
}
</style>
"""
