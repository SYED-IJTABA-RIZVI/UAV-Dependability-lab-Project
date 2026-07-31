"""CSS lifted from sky_detection_console_final.html, adapted for Streamlit's DOM."""

CSS = """
<style>
:root{
  --bg:#F4F5F6;
  --panel:#FFFFFF;
  --panel-alt:#F8F9FA;
  --border:#DFE2E5;
  --border-strong:#C2C7CC;
  --text-hi:#1B2126;
  --text-mid:#616B74;
  --text-low:#8D959D;
  --accent:#2C5A7C;
  --accent-bg:#E8EEF2;
  --status-safe:#2E7D4F;
  --status-safe-bg:#E7F3EB;
  --status-caution:#9C6B0E;
  --status-caution-bg:#F7F0DE;
  --status-threat:#B23A31;
  --status-threat-bg:#F6E6E4;
  --radius:4px;
  --mono:'JetBrains Mono','Consolas',monospace;
}

.stApp{ background:var(--bg); }
[data-testid="stSidebar"]{ display:none; }

.sdt-header{
  display:flex; align-items:center; justify-content:space-between;
  background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
  padding:11px 20px; margin-bottom:14px;
}
.sdt-brand{ display:flex; align-items:center; gap:10px; }
.sdt-logo{ width:30px; height:30px; border-radius:var(--radius); background:var(--accent);
  display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700; font-size:11.5px; }
.sdt-brand-name{ font-size:14px; font-weight:600; color:var(--text-hi); }
.sdt-brand-sub{ font-size:10.5px; color:var(--text-low); font-family:var(--mono); }
.sdt-header-right{ display:flex; align-items:center; gap:14px; }
.sdt-classes{ display:flex; align-items:center; gap:7px; font-size:11px; color:var(--text-mid); }
.class-pill{ font-weight:600; font-size:10.5px; padding:3px 9px; border-radius:var(--radius);
  border:1px solid var(--border-strong); color:var(--text-mid); font-family:var(--mono); }
.status-pill{ display:flex; align-items:center; gap:6px; font-size:10.5px; color:var(--status-safe);
  border:1px solid var(--status-safe); padding:5px 12px; border-radius:var(--radius);
  font-family:var(--mono); background:var(--status-safe-bg); white-space:nowrap; }
.status-dot{ width:6px; height:6px; border-radius:50%; background:var(--status-safe); display:inline-block; }

.sdt-panel{ background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
  padding:16px; box-shadow:0 1px 2px rgba(20,25,32,0.05); height:100%; }
.panel-title{ font-size:12.5px; font-weight:700; color:var(--text-hi); }
.panel-sub{ font-size:11px; color:var(--text-low); margin-top:3px; line-height:1.5; margin-bottom:12px; }
.section-label{ font-size:10px; font-weight:700; letter-spacing:0.5px; color:var(--text-low);
  text-transform:uppercase; margin:16px 0 8px; font-family:var(--mono); }

.mode-card{ border:1.5px solid var(--border); border-radius:var(--radius); padding:11px 12px;
  margin-top:8px; }
.mode-card.selected{ border-color:var(--accent); background:var(--accent-bg); }
.mode-head{ display:flex; justify-content:space-between; align-items:center; }
.mode-name{ font-size:12px; font-weight:700; color:var(--text-hi); }
.mode-badge{ font-size:9px; font-weight:700; padding:2px 8px; border-radius:2px; border:1px solid var(--accent);
  color:var(--accent); font-family:var(--mono); }
.mode-desc{ font-size:10.5px; color:var(--text-mid); margin-top:4px; line-height:1.5; }

.queue-item{ display:flex; gap:10px; align-items:center; padding:9px; border-radius:var(--radius);
  border:1px solid var(--border); background:var(--panel-alt); margin-top:8px; }
.queue-thumb{ width:30px; height:30px; border-radius:2px; background:var(--border-strong); flex-shrink:0; }
.queue-name{ font-size:11px; font-weight:600; color:var(--text-hi); }
.queue-meta{ font-size:9.5px; color:var(--text-low); font-family:var(--mono); }
.queue-status{ margin-left:auto; font-family:var(--mono); font-size:9px; color:var(--status-safe); }

.ws-eyebrow{ font-size:10px; font-weight:700; color:var(--accent); letter-spacing:0.6px; font-family:var(--mono); }
.ws-title{ font-size:14.5px; font-weight:600; margin-top:2px; color:var(--text-hi); }

.hud-chip{ font-family:var(--mono); font-size:10.5px; background:rgba(0,0,0,0.55); color:#DCE4EC;
  padding:4px 9px; border-radius:2px; display:inline-block; }

.metric-box{ margin-top:14px; border:1px solid var(--border); border-radius:var(--radius); padding:11px 12px;
  background:var(--panel-alt); }
.metric-title{ font-size:10px; font-weight:700; color:var(--text-mid); letter-spacing:0.4px; font-family:var(--mono); }
.latency-row{ display:flex; justify-content:space-between; margin-top:10px; font-family:var(--mono); font-size:10.5px; }
.latency-item{ display:flex; flex-direction:column; gap:2px; }
.latency-label{ color:var(--text-low); font-size:9px; text-transform:uppercase; letter-spacing:0.3px; }
.latency-val{ font-weight:700; color:var(--accent); }

.vlm-panel{ margin-top:14px; background:var(--panel-alt); border:1px solid var(--accent);
  border-radius:var(--radius); padding:12px 13px; }
.vlm-panel-head{ display:flex; align-items:center; justify-content:space-between; }
.vlm-panel-title{ font-size:10.5px; font-weight:700; letter-spacing:0.5px; color:var(--accent); font-family:var(--mono); }
.vlm-panel-lat{ font-family:var(--mono); font-size:10.5px; color:var(--accent); }
.vlm-panel-sub{ font-size:9.5px; color:var(--text-low); margin-top:5px; font-family:var(--mono); }
.vlm-panel-cls{ font-size:14px; font-weight:700; margin-top:9px; color:var(--text-hi); }
.vlm-panel-conf{ font-family:var(--mono); font-size:11px; font-weight:400; color:var(--accent); }
.vlm-panel-reason{ font-size:11px; color:var(--text-mid); margin-top:7px; line-height:1.5; }

.ws-footer{ font-size:11px; color:var(--text-low); font-family:var(--mono); margin-top:10px; padding-top:10px;
  border-top:1px solid var(--border); }

.final-badge-threat{ color:var(--status-threat); border:1px solid var(--status-threat);
  background:var(--status-threat-bg); padding:3px 10px; border-radius:2px; font-family:var(--mono); font-size:10.5px; font-weight:700; }
.final-badge-safe{ color:var(--status-safe); border:1px solid var(--status-safe);
  background:var(--status-safe-bg); padding:3px 10px; border-radius:2px; font-family:var(--mono); font-size:10.5px; font-weight:700; }
.final-badge-caution{ color:var(--status-caution); border:1px solid var(--status-caution);
  background:var(--status-caution-bg); padding:3px 10px; border-radius:2px; font-family:var(--mono); font-size:10.5px; font-weight:700; }

.custom-model-item{ display:flex; align-items:center; justify-content:space-between;
  padding:9px 11px; border:1px solid var(--border); border-radius:2px; background:var(--panel-alt); }
.cmi-name{ font-family:var(--mono); font-size:11px; color:var(--text-hi); font-weight:600; }
.cmi-meta{ font-family:var(--mono); font-size:9px; color:var(--text-low); margin-top:2px; }
.cmi-badge{ font-family:var(--mono); font-size:9px; padding:2px 8px; border-radius:2px; color:var(--status-safe);
  border:1px solid var(--status-safe); background:var(--status-safe-bg); white-space:nowrap; }

/* Streamlit widget text renders near-invisible against the light panel background
   under some themes — force it dark and legible everywhere inside a panel. */
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

/* Some widget chrome keeps its own dark background from the underlying theme —
   forcing dark text on top of that makes it unreadable. Give those pieces an
   explicit light background to match the panel instead of just recoloring text. */
[data-testid="stFileUploaderDropzone"]{
  background:var(--panel-alt) !important; border:1.5px dashed var(--border-strong) !important;
}
[data-testid="stFileUploaderDropzone"] button{
  background:#fff !important; color:var(--text-hi) !important; border:1px solid var(--border-strong) !important;
}
[data-testid="stSelectbox"] > div > div{
  background:#fff !important; border-color:var(--border-strong) !important;
}
div[data-baseweb="popover"] *{
  color:var(--text-hi) !important;
}
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"]{
  background:#fff !important;
}
div[data-baseweb="popover"] li:hover{
  background:var(--accent-bg) !important;
}
</style>
"""
