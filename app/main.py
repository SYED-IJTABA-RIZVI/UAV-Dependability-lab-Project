import io
from pathlib import Path

import streamlit as st
from PIL import Image

import batch
import db
from drawing import draw_multi
from mock_backend import CLASS_COLOR, ModelUnavailable, run_cascade
from style import CSS

DEFAULT_SINGLE_OUTPUT_DIR = "outputs/single_images"

st.set_page_config(page_title="Sky Object Detection Tool", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

RFDETR_MODELS = ["YOLOv12-RGB (built-in)", "YOLOv10-IR / Thermal (built-in)"]
VLM_MODELS = ["InternVL2.5", "DeepSeek-VL", "Qwen2.5-VL", "BLIP-2"]
MODES = [
    ("YOLO Only", "EDGE-FAST", "Real-time detection only. Low latency; no fallback if confidence is low."),
    ("VLM Only", "DEEP-VLM", "Multi-modal reasoning on every image. Higher contextual accuracy, slower inference."),
    ("YOLO & VLM (Adaptive Fallback)", "ADAPTIVE FALLBACK",
     "YOLO classifies instantly. Below the confidence cutoff, the VLM re-checks and becomes the source of truth."),
]


@st.cache_resource
def _init_db_once() -> bool:
    return db.init_db()


db_ready = _init_db_once()

for key, default in [
    ("result", None), ("image_bytes", None), ("filename", None),
    ("folder_result", None), ("eval_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- HEADER ----------
status_text = "DB CONNECTED &middot; MODEL NOT YET LOADED" if db_ready else "DB OFFLINE &middot; RUNNING ON LOCAL FILES ONLY"
st.markdown(
    f"""
    <div class="sdt-header">
      <div class="sdt-brand">
        <div class="sdt-logo">SDT</div>
        <div>
          <div class="sdt-brand-name">Sky Object Detection Tool</div>
          <div class="sdt-brand-sub">YOLO / VLM CASCADE &mdash; RESEARCH PROTOTYPE</div>
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
        <div class="status-pill"><span class="status-dot"></span>{status_text}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, center, right = st.columns([1.2, 2.2, 1.3], gap="medium")

# ---------- LEFT (part 1): INPUT SOURCE ----------
with left:
    st.markdown('<div class="sdt-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">&uarr; INPUT SOURCE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-sub">Upload a single image, batch-process a folder, or run a full '
        'evaluation against a labeled test folder.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="source-tabs">', unsafe_allow_html=True)
    tab = st.radio("Source", ["IMAGE", "FOLDER", "TEST FOLDER", "HISTORY"], horizontal=True,
                    label_visibility="collapsed", key="source_tab")
    st.markdown("</div>", unsafe_allow_html=True)

    if tab != "HISTORY":
        st.markdown('<div class="section-label">MODALITY</div>', unsafe_allow_html=True)
        modality = st.radio("Modality", ["RGB", "IR / THERMAL"], horizontal=True, label_visibility="collapsed")
    else:
        modality = "RGB"

    if tab == "IMAGE":
        uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"],
                                     label_visibility="collapsed")
        if uploaded is not None:
            st.session_state.image_bytes = uploaded.getvalue()
            st.session_state.filename = uploaded.name
            st.markdown(
                f"""
                <div class="section-label">QUEUE (1)</div>
                <div class="queue-item">
                  <div>
                    <div class="queue-name">{uploaded.name}</div>
                    <div class="queue-meta">{len(st.session_state.image_bytes) // 1024} KB &middot; {modality}</div>
                  </div>
                  <div class="queue-status">READY</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('<div class="section-label">OUTPUT LOCATION PATH (optional)</div>', unsafe_allow_html=True)
        single_output_path = st.text_input(
            "Output path", placeholder=f"default: {DEFAULT_SINGLE_OUTPUT_DIR}",
            label_visibility="collapsed", key="single_output",
        )

    elif tab == "FOLDER":
        st.markdown('<div class="section-label">IMAGE FOLDER PATH</div>', unsafe_allow_html=True)
        folder_path = st.text_input("Folder path", placeholder="/path/to/images", label_visibility="collapsed")
        st.markdown('<div class="section-label">OUTPUT LOCATION PATH</div>', unsafe_allow_html=True)
        folder_output_path = st.text_input("Output path", placeholder="/path/to/output",
                                            label_visibility="collapsed", key="folder_output")

    elif tab == "TEST FOLDER":
        st.markdown('<div class="section-label">TEST FOLDER PATH</div>', unsafe_allow_html=True)
        test_folder_path = st.text_input("Test folder path", placeholder="/path/to/test_folder",
                                          label_visibility="collapsed")
        st.caption("Must contain images/ and labels/ (YOLO .txt) subfolders.")
        st.markdown('<div class="section-label">OUTPUT LOCATION PATH</div>', unsafe_allow_html=True)
        eval_output_path = st.text_input("Output path", placeholder="/path/to/output",
                                          label_visibility="collapsed", key="eval_output")
        st.markdown('<div class="section-label">MATCH IOU THRESHOLD</div>', unsafe_allow_html=True)
        iou_threshold = st.slider("IoU threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                                   label_visibility="collapsed")

    else:  # HISTORY
        st.markdown(
            '<div class="panel-sub">Browse past runs (single image, folder, test-folder eval) '
            'saved to Postgres.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT (part 1): DETECTION PIPELINE ----------
mode, rfdetr_model, vlm_model, threshold, analyze_disabled = None, None, None, None, True

with right:
  if tab == "HISTORY":
    st.markdown('<div class="sdt-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">&#9670; HISTORY</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-sub">Every single-image, folder, and test-folder run is persisted '
        'to Postgres — pick one on the left to inspect it.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
  else:
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

    st.markdown('<div class="section-label">YOLO DETECTION MODEL</div>', unsafe_allow_html=True)
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
    st.caption("Requires the YOLO checkpoint + service running (see SETUP.md) — no fallback if unavailable.")

    st.markdown('<div class="section-label">VISION-LANGUAGE MODEL (VLM)</div>', unsafe_allow_html=True)
    vlm_model = st.selectbox("vlm model", VLM_MODELS, label_visibility="collapsed",
                              disabled=(mode == "YOLO Only"))
    st.caption("Requires an API key / local service for this VLM (see SETUP.md) — no fallback if unavailable.")

    st.markdown('<div class="section-label">CONFIDENCE CUTOFF</div>', unsafe_allow_html=True)
    threshold = st.slider("threshold", min_value=0.15, max_value=0.90, value=0.39, step=0.01,
                           label_visibility="collapsed")

    analyze_disabled = st.session_state.image_bytes is None
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- LEFT (part 2): RUN buttons for FOLDER / TEST FOLDER ----------
with left:
    if tab == "FOLDER":
        st.markdown('<div class="sdt-panel" style="margin-top:14px;">', unsafe_allow_html=True)
        run_disabled = not (folder_path and folder_output_path)
        if st.button("▶ RUN FOLDER BATCH", disabled=run_disabled, type="primary", use_container_width=True):
            with st.spinner("Processing folder..."):
                try:
                    st.session_state.folder_result = batch.run_folder(
                        folder_path, folder_output_path, modality, rfdetr_model, vlm_model, threshold, mode
                    )
                except Exception as exc:
                    st.error(f"Folder run failed: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

    elif tab == "TEST FOLDER":
        st.markdown('<div class="sdt-panel" style="margin-top:14px;">', unsafe_allow_html=True)
        run_disabled = not (test_folder_path and eval_output_path)
        if st.button("▶ RUN EVALUATION", disabled=run_disabled, type="primary", use_container_width=True):
            with st.spinner("Running evaluation..."):
                try:
                    st.session_state.eval_result = batch.run_test_folder(
                        test_folder_path, eval_output_path, modality, rfdetr_model, vlm_model,
                        threshold, mode, iou_threshold
                    )
                except Exception as exc:
                    st.error(f"Evaluation run failed: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- CENTER: WORKSPACE ----------
with center:
    st.markdown('<div class="sdt-panel">', unsafe_allow_html=True)

    if tab == "IMAGE":
        title = st.session_state.filename or "No image selected"
        st.markdown(
            f'<div class="ws-eyebrow">ACTIVE WORKSPACE</div><div class="ws-title">{title}</div>',
            unsafe_allow_html=True,
        )

        analyze = st.button("▶ ANALYZE FRAME", disabled=analyze_disabled, type="primary")

        if analyze:
            spinner_msg = ("Analyzing — if this VLM wasn't the last one used, it's loading fresh "
                            "into VRAM now, which can take a minute...")
            try:
                with st.spinner(spinner_msg):
                    st.session_state.result = run_cascade(
                        st.session_state.image_bytes, modality, rfdetr_model, vlm_model, threshold, mode
                    )
            except ModelUnavailable as exc:
                st.session_state.result = None
                st.error(f"MODEL UNAVAILABLE — {exc}")
            else:
                res = st.session_state.result

                out_dir = Path(single_output_path) if single_output_path else Path(DEFAULT_SINGLE_OUTPUT_DIR)
                out_dir.mkdir(parents=True, exist_ok=True)
                annotated_img = Image.open(io.BytesIO(st.session_state.image_bytes))
                if res.get("rfdetr_all"):
                    annotated_img = draw_multi(annotated_img, res["rfdetr_all"], CLASS_COLOR)
                annotated_path = out_dir / st.session_state.filename
                annotated_img.convert("RGB").save(annotated_path)

                db.save_single_image_run(st.session_state.filename, modality, rfdetr_model, vlm_model,
                                          threshold, mode, res,
                                          output_path=str(out_dir), annotated_path=str(annotated_path))

        result = st.session_state.result

        if st.session_state.image_bytes is not None:
            img = Image.open(io.BytesIO(st.session_state.image_bytes))
            if result is not None and result.get("rfdetr_all"):
                img = draw_multi(img, result["rfdetr_all"], CLASS_COLOR)
            st.image(img, use_container_width=True)

            if result is not None and result.get("rfdetr_all"):
                iw, ih = img.size
                lines = [f"Image size: {iw} x {ih}", f"YOLO detections: {len(result['rfdetr_all'])}"]
                for i, r in enumerate(result["rfdetr_all"], start=1):
                    bx, by, bw, bh = r["bbox"]
                    lines.append(
                        f"  [{i}] {r['class_name']} · {r['confidence']:.2f} · "
                        f"fractional=({bx:.4f}, {by:.4f}, {bw:.4f}, {bh:.4f}) · "
                        f"pixels=({bx*iw:.1f}, {by*ih:.1f}, {(bx+bw)*iw:.1f}, {(by+bh)*ih:.1f})"
                    )
                st.code("\n".join(lines), language="text")
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
                footer += " &middot; ESCALATED TO VLM (YOLO CONFIDENCE BELOW CUTOFF)"
            st.markdown(f'<div class="ws-footer">{footer}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ws-footer">NO ANALYSIS RUN YET ON THIS IMAGE.</div>', unsafe_allow_html=True)

    elif tab == "FOLDER":
        st.markdown('<div class="ws-eyebrow">ACTIVE WORKSPACE</div><div class="ws-title">Folder Batch</div>',
                    unsafe_allow_html=True)

        fr = st.session_state.folder_result
        if fr is None:
            st.markdown(
                '<div style="padding:60px; text-align:center; color:var(--text-low); '
                'font-family:var(--mono); background:var(--panel-alt); border-radius:4px;">'
                'SET A FOLDER + OUTPUT PATH AND RUN THE BATCH</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="ws-footer">{fr["num_images"]} IMAGES PROCESSED &middot; '
                f'{fr["num_cascaded"]} ESCALATED TO VLM &middot; OUTPUT: {fr["output_path"]}</div>',
                unsafe_allow_html=True,
            )
            if fr["images"]:
                names = [im["filename"] for im in fr["images"]]
                picked = st.selectbox("Preview image", names, label_visibility="collapsed")
                path = next(im["annotated_path"] for im in fr["images"] if im["filename"] == picked)
                st.image(Image.open(path), use_container_width=True)

    elif tab == "TEST FOLDER":
        st.markdown('<div class="ws-eyebrow">ACTIVE WORKSPACE</div><div class="ws-title">Test Folder Evaluation</div>',
                    unsafe_allow_html=True)

        er = st.session_state.eval_result
        if er is None:
            st.markdown(
                '<div style="padding:60px; text-align:center; color:var(--text-low); '
                'font-family:var(--mono); background:var(--panel-alt); border-radius:4px;">'
                'SET A TEST FOLDER + OUTPUT PATH AND RUN THE EVALUATION</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="ws-footer">{er["num_images"]} IMAGES EVALUATED &middot; OUTPUT: {er["output_path"]}</div>',
                unsafe_allow_html=True,
            )
            if er["images"]:
                names = [im["filename"] for im in er["images"]]
                picked = st.selectbox("Preview image", names, label_visibility="collapsed")
                path = next(im["annotated_path"] for im in er["images"] if im["filename"] == picked)
                st.image(Image.open(path), use_container_width=True)
                st.caption("Dashed gray = ground truth · solid color = prediction (tagged YOLO/VLM) "
                           "· red UNDETECTED = ground truth the pipeline never matched.")

    else:  # HISTORY
        st.markdown('<div class="ws-eyebrow">ACTIVE WORKSPACE</div><div class="ws-title">Run History</div>',
                    unsafe_allow_html=True)

        if not db_ready:
            st.markdown(
                '<div style="padding:60px; text-align:center; color:var(--text-low); '
                'font-family:var(--mono); background:var(--panel-alt); border-radius:4px;">'
                'DB OFFLINE &mdash; NO HISTORY AVAILABLE</div>',
                unsafe_allow_html=True,
            )
        else:
            runs = db.list_runs()
            if not runs:
                st.markdown(
                    '<div style="padding:60px; text-align:center; color:var(--text-low); '
                    'font-family:var(--mono); background:var(--panel-alt); border-radius:4px;">'
                    'NO RUNS RECORDED YET</div>',
                    unsafe_allow_html=True,
                )
            else:
                run_labels = {
                    r["id"]: (f"#{r['id']} · {r['created_at']:%Y-%m-%d %H:%M} · {r['mode'].upper()} · "
                               f"{r['num_images']} img · {r['pipeline_mode']}")
                    for r in runs
                }
                picked_id = st.selectbox("Pick a run", list(run_labels.keys()),
                                          format_func=lambda i: run_labels[i], label_visibility="collapsed")
                detail = db.get_run_detail(picked_id)
                st.session_state.history_detail = detail

                if detail:
                    st.markdown(
                        f'<div class="ws-footer">MODE: {detail["mode"].upper()} &middot; '
                        f'YOLO: {detail["rfdetr_model"]} &middot; VLM: {detail["vlm_model"]} &middot; '
                        f'CUTOFF: {detail["confidence_cutoff"]:.2f} &middot; '
                        f'SOURCE: {detail["source_path"] or "—"} &middot; '
                        f'OUTPUT: {detail["output_path"] or "—"}</div>',
                        unsafe_allow_html=True,
                    )
                    if detail["images"]:
                        names = [im["filename"] for im in detail["images"]]
                        picked_img = st.selectbox("Preview image", names, label_visibility="collapsed",
                                                   key="history_img_pick")
                        rec = next(im for im in detail["images"] if im["filename"] == picked_img)
                        if rec["path"] and Path(rec["path"]).exists():
                            st.image(Image.open(rec["path"]), use_container_width=True)
                        else:
                            st.caption("Annotated image not found on disk (path may be from another machine).")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT (part 2): per-tab results ----------
with right:
    if tab == "IMAGE":
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
                    <div class="latency-item"><span class="latency-label">YOLO PASS</span><span class="latency-val">{rfdetr_lat}</span></div>
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

    elif tab == "FOLDER":
        fr = st.session_state.folder_result
        if fr is not None:
            st.markdown('<div class="sdt-panel" style="margin-top:14px;">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">RESULTS</div>', unsafe_allow_html=True)
            with open(fr["results_csv"], "rb") as f:
                st.download_button("DOWNLOAD results.csv", f, file_name="results.csv", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    elif tab == "TEST FOLDER":
        er = st.session_state.eval_result
        if er is not None:
            st.markdown('<div class="sdt-panel" style="margin-top:14px;">', unsafe_allow_html=True)

            combined = er["scope_metrics"]["combined"]
            st.markdown(
                f"""
                <div class="metric-box" style="margin-top:0;">
                  <div class="metric-title">COMBINED PIPELINE METRICS</div>
                  <div class="latency-row">
                    <div class="latency-item"><span class="latency-label">PRECISION</span><span class="latency-val">{combined['precision']:.2f}</span></div>
                    <div class="latency-item"><span class="latency-label">RECALL</span><span class="latency-val">{combined['recall']:.2f}</span></div>
                    <div class="latency-item"><span class="latency-label">F1</span><span class="latency-val">{combined['f1']:.2f}</span></div>
                  </div>
                  <div class="latency-row">
                    <div class="latency-item"><span class="latency-label">ACCURACY</span><span class="latency-val">{combined['accuracy']:.2f}</span></div>
                    <div class="latency-item"><span class="latency-label">MEAN IOU</span><span class="latency-val">{combined['mean_iou']:.2f}</span></div>
                    <div class="latency-item"><span class="latency-label">RATE</span><span class="latency-val">{combined['images_per_sec']:.1f}/s</span></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-label">SCOPE COMPARISON</div>', unsafe_allow_html=True)
            scope_rows = []
            for scope_key, scope_name in [("rfdetr", "YOLO"), ("vlm", "VLM"), ("combined", "Combined")]:
                m = er["scope_metrics"][scope_key]
                scope_rows.append({
                    "Scope": scope_name, "Precision": round(m["precision"], 2), "Recall": round(m["recall"], 2),
                    "F1": round(m["f1"], 2), "Accuracy": round(m["accuracy"], 2),
                    "Mean IoU": round(m["mean_iou"], 2), "Avg time/img (ms)": round(m["avg_time_per_image_ms"], 1),
                })
            st.dataframe(scope_rows, hide_index=True, use_container_width=True)

            st.markdown('<div class="section-label">UNDETECTED &mdash; BY CLASS (% OF UNDETECTED)</div>',
                         unsafe_allow_html=True)
            if er["undetected_breakdown"]:
                st.bar_chart(er["undetected_breakdown"])
            else:
                st.caption("No undetected ground-truth objects in this run.")

            st.markdown('<div class="section-label">CONFUSION MATRICES</div>', unsafe_allow_html=True)
            hm_tabs = st.tabs(["YOLO", "VLM", "Combined", "IoU dist."])
            for hm_tab, key in zip(hm_tabs, ["rfdetr", "vlm", "combined", "iou_distribution"]):
                with hm_tab:
                    st.image(er["heatmap_paths"][key], use_container_width=True)

            st.markdown('<div class="section-label">DOWNLOADS</div>', unsafe_allow_html=True)
            for label, path in [
                ("results.csv (per-detection)", er["csv_paths"]["results"]),
                ("images_summary.csv", er["csv_paths"]["images_summary"]),
                ("metrics_summary.csv", er["csv_paths"]["metrics_summary"]),
                ("undetected_by_class.csv", er["csv_paths"]["undetected_by_class"]),
            ]:
                with open(path, "rb") as f:
                    st.download_button(f"DOWNLOAD {label}", f, file_name=path.split("/")[-1],
                                        use_container_width=True, key=f"dl_{label}")

            st.markdown("</div>", unsafe_allow_html=True)

    else:  # HISTORY
        detail = st.session_state.get("history_detail")
        if detail is not None:
            st.markdown('<div class="sdt-panel" style="margin-top:14px;">', unsafe_allow_html=True)

            if detail["metrics"]:
                st.markdown('<div class="section-label">SCOPE METRICS</div>', unsafe_allow_html=True)
                scope_rows = [
                    {"Scope": m["scope"].capitalize(), "Precision": round(m["precision"], 2),
                     "Recall": round(m["recall"], 2), "F1": round(m["f1"], 2),
                     "Accuracy": round(m["accuracy"], 2), "Mean IoU": round(m["mean_iou"], 2),
                     "Avg time/img (ms)": round(m["avg_time_per_image_ms"], 1)}
                    for m in detail["metrics"]
                ]
                st.dataframe(scope_rows, hide_index=True, use_container_width=True)

            if detail["undetected"]:
                st.markdown('<div class="section-label">UNDETECTED &mdash; BY CLASS (% OF UNDETECTED)</div>',
                             unsafe_allow_html=True)
                st.bar_chart({u["class_name"]: u["percentage"] for u in detail["undetected"]})

            st.markdown('<div class="section-label">DETECTIONS (SELECTED IMAGE)</div>', unsafe_allow_html=True)
            picked_img = st.session_state.get("history_img_pick")
            rec = next((im for im in detail["images"] if im["filename"] == picked_img), None)
            if rec and rec["detections"]:
                det_rows = [
                    {"Source": d["source"], "Class": d["class_name"], "Confidence": round(d["confidence"], 2),
                     "Cascaded": d["cascaded"], "IoU": round(d["iou"], 2) if d["iou"] is not None else "",
                     "Match": d["match_result"] or "", "Latency (ms)": d["latency_ms"],
                     "VLM reasoning": d["reasoning"] or ""}
                    for d in rec["detections"]
                ]
                st.dataframe(det_rows, hide_index=True, use_container_width=True)
            else:
                st.caption("No detections recorded for this image.")

            st.markdown("</div>", unsafe_allow_html=True)
