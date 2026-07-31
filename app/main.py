import io

import streamlit as st
from PIL import Image, ImageDraw

from mock_backend import CLASSES, run_cascade
from style import CSS

st.set_page_config(page_title="Sky Object Detection Tool", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

RFDETR_MODELS = ["RFDETR-RGB (built-in)", "RFDETR-IR / Thermal (built-in)"]
VLM_MODELS = ["InternVL3", "DeepSeek-VL", "Qwen2.5-VL", "BLIP-2"]
MODES = [
    ("RFDETR Only", "EDGE-FAST", "Real-time detection only. Low latency; no fallback if confidence is low."),
    ("VLM Only", "DEEP-VLM", "Multi-modal reasoning on every image. Higher contextual accuracy, slower inference."),
    ("RFDETR & VLM (Adaptive Fallback)", "ADAPTIVE FALLBACK",
     "RFDETR classifies instantly. Below the confidence cutoff, the VLM re-checks and becomes the source of truth."),
]

CLASS_COLOR = {
    "Drone": "#B23A31",
    "Bird": "#9C6B0E",
    "Helicopter": "#B23A31",
    "Airplane": "#2C5A7C",
}

if "result" not in st.session_state:
    st.session_state.result = None
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None
if "filename" not in st.session_state:
    st.session_state.filename = None


def draw_bbox(image: Image.Image, bbox, label: str, color: str) -> Image.Image:
    img = image.convert("RGB").copy()
    w, h = img.size
    x, y, bw, bh = bbox
    px, py, pw, ph = x * w, y * h, bw * w, bh * h

    draw = ImageDraw.Draw(img)
    draw.rectangle([px, py, px + pw, py + ph], outline=color, width=3)
    text_y = max(py - 20, 0)
    draw.rectangle([px, text_y, px + max(len(label) * 7 + 12, 60), text_y + 18], fill=color)
    draw.text((px + 6, text_y + 3), label, fill="white")
    return img


# ---------- HEADER ----------
st.markdown(
    """
    <div class="sdt-header">
      <div class="sdt-brand">
        <div class="sdt-logo">SDT</div>
        <div>
          <div class="sdt-brand-name">Sky Object Detection Tool</div>
          <div class="sdt-brand-sub">RFDETR / VLM CASCADE &mdash; RESEARCH PROTOTYPE</div>
        </div>
      </div>
      <div class="sdt-header-right">
        <div class="sdt-classes">
          ACTIVE CLASSES:
          <span class="class-pill">DRONE</span>
          <span class="class-pill">BIRD</span>
          <span class="class-pill">HELICOPTER</span>
          <span class="class-pill">AIRPLANE</span>
        </div>
        <div class="status-pill"><span class="status-dot"></span>MOCK BACKEND &middot; MODEL NOT YET LOADED</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, center, right = st.columns([1.1, 2.3, 1.3], gap="medium")

# ---------- LEFT: INPUT SOURCE ----------
with left:
    st.markdown('<div class="sdt-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">&uarr; INPUT SOURCE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-sub">Upload a single RGB or thermal image for classification.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">MODALITY</div>', unsafe_allow_html=True)
    modality = st.radio("Modality", ["RGB", "IR / THERMAL"], horizontal=True, label_visibility="collapsed")

    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")

    if uploaded is not None:
        st.session_state.image_bytes = uploaded.getvalue()
        st.session_state.filename = uploaded.name
        st.markdown(
            f"""
            <div class="section-label">QUEUE (1)</div>
            <div class="queue-item">
              <div class="queue-thumb"></div>
              <div>
                <div class="queue-name">{uploaded.name}</div>
                <div class="queue-meta">{len(st.session_state.image_bytes) // 1024} KB &middot; {modality}</div>
              </div>
              <div class="queue-status">READY</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT: DETECTION PIPELINE ----------
with right:
    st.markdown('<div class="sdt-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">&#9670; DETECTION PIPELINE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-sub">Configure detection models and fallback behavior.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">PIPELINE ANALYTICS MODE</div>', unsafe_allow_html=True)
    mode_labels = [m[0] for m in MODES]
    mode = st.radio("Pipeline analytics mode", mode_labels, index=2, label_visibility="collapsed")
    for name, badge, desc in MODES:
        cls = "mode-card selected" if name == mode else "mode-card"
        st.markdown(
            f"""
            <div class="{cls}">
              <div class="mode-head"><span class="mode-name">{name}</span><span class="mode-badge">{badge}</span></div>
              <div class="mode-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">RFDETR DETECTION MODEL</div>', unsafe_allow_html=True)
    rfdetr_model = RFDETR_MODELS[0] if modality == "RGB" else RFDETR_MODELS[1]
    dim_style = "opacity:0.5;" if mode == "VLM Only" else ""
    st.markdown(
        f"""
        <div class="custom-model-item" style="{dim_style}">
          <div><div class="cmi-name">{rfdetr_model}</div>
          <div class="cmi-meta">AUTO-SELECTED FROM MODALITY: {modality}</div></div>
          <div class="cmi-badge">ACTIVE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Trained checkpoint lives on the lab PC — not loaded in this prototype.")

    st.markdown('<div class="section-label">VISION-LANGUAGE MODEL (VLM)</div>', unsafe_allow_html=True)
    vlm_model = st.selectbox("vlm model", VLM_MODELS, label_visibility="collapsed",
                              disabled=(mode == "RFDETR Only"))
    st.caption("API key not yet configured — fallback calls are mocked.")

    st.markdown('<div class="section-label">CONFIDENCE CUTOFF</div>', unsafe_allow_html=True)
    threshold = st.slider("threshold", min_value=0.15, max_value=0.90, value=0.39, step=0.01,
                           label_visibility="collapsed")

    analyze_disabled = st.session_state.image_bytes is None
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- CENTER: WORKSPACE ----------
with center:
    st.markdown('<div class="sdt-panel">', unsafe_allow_html=True)
    title = st.session_state.filename or "No image selected"
    st.markdown(
        f"""
        <div class="ws-eyebrow">ACTIVE WORKSPACE</div>
        <div class="ws-title">{title}</div>
        """,
        unsafe_allow_html=True,
    )

    analyze = st.button("▶ ANALYZE FRAME", disabled=analyze_disabled, type="primary")

    if analyze:
        st.session_state.result = run_cascade(
            st.session_state.image_bytes, modality, rfdetr_model, vlm_model, threshold, mode
        )

    result = st.session_state.result

    if st.session_state.image_bytes is not None:
        img = Image.open(io.BytesIO(st.session_state.image_bytes))
        if result is not None and result.get("rfdetr") is not None:
            r = result["rfdetr"]
            label = f"{r['class_name']} · {r['confidence']:.2f} · RFDETR"
            img = draw_bbox(img, r["bbox"], label, CLASS_COLOR.get(r["class_name"], "#2C5A7C"))
        st.image(img, use_container_width=True)
    else:
        st.markdown(
            '<div style="padding:60px; text-align:center; color:var(--text-low); '
            'font-family:var(--mono); background:var(--panel-alt); border-radius:4px;">'
            'UPLOAD AN IMAGE TO BEGIN</div>',
            unsafe_allow_html=True,
        )

    if result is not None:
        final = result["final"]
        footer = (
            f"FINAL SOURCE OF TRUTH: {final['source']} &middot; {final['class_name']} "
            f"&middot; {final['confidence']:.2f}"
        )
        if result["cascaded"]:
            footer += " &middot; ESCALATED TO VLM (RFDETR CONFIDENCE BELOW CUTOFF)"
        st.markdown(f'<div class="ws-footer">{footer}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="ws-footer">NO ANALYSIS RUN YET ON THIS IMAGE.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT (continued): metrics + VLM output ----------
with right:
    result = st.session_state.result
    if result is not None:
        st.markdown('<div class="sdt-panel" style="margin-top:14px;">', unsafe_allow_html=True)

        rfdetr_lat = f"{result['rfdetr']['latency_ms']} ms" if result["rfdetr"] else "—"
        vlm_lat = f"{result['vlm']['latency_ms']} ms" if result["vlm"] else "—"
        cascade_rate = "TRIGGERED" if result["cascaded"] else "NOT TRIGGERED"

        st.markdown(
            f"""
            <div class="metric-box" style="margin-top:0;">
              <div class="metric-title">INFERENCE LATENCY (MOCK)</div>
              <div class="latency-row">
                <div class="latency-item"><span class="latency-label">RFDETR PASS</span><span class="latency-val">{rfdetr_lat}</span></div>
                <div class="latency-item"><span class="latency-label">VLM FALLBACK</span><span class="latency-val">{vlm_lat}</span></div>
                <div class="latency-item"><span class="latency-label">CASCADE</span><span class="latency-val">{cascade_rate}</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result["vlm"] is not None:
            v = result["vlm"]
            sub = (
                f"TRIGGERED &middot; CONFIDENCE {result['rfdetr']['confidence']:.2f} BELOW CUTOFF {threshold:.2f}"
                if result["cascaded"] else "VLM-ONLY MODE"
            )
            st.markdown(
                f"""
                <div class="vlm-panel">
                  <div class="vlm-panel-head">
                    <div class="vlm-panel-title">&#9670; VLM OUTPUT &mdash; {v['model_name']}</div>
                    <div class="vlm-panel-lat">{v['latency_ms']} ms</div>
                  </div>
                  <div class="vlm-panel-sub">{sub}</div>
                  <div class="vlm-panel-cls">{v['class_name']} <span class="vlm-panel-conf">{v['confidence']:.2f}</span></div>
                  <div class="vlm-panel-reason">{v['reasoning']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)
