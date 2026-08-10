"""
FastAPI wrapper around Ultralytics YOLO, serving the lab's two trained
checkpoints: YOLOv12 for RGB, YOLOv10 for IR/Thermal. Replaces the earlier
RFDETR-based detector (see git history / SETUP.md for context on the switch).

Class names are read directly from each checkpoint's embedded metadata
(`result.names`, set by Ultralytics at training/export time) rather than a
hardcoded list here — this avoids the class-order mismatch bug the RFDETR
version had (checkpoint's real training order didn't match a manually
maintained CLASSES list). If a checkpoint's class names don't match this
app's expected set (Drone/Bird/Helicopter/Airplane, case-sensitive), the
mismatch will surface as real_inference.py raising "unknown class" and
falling back to the mock simulator — check this service's /health or logs
if that happens.

NOT verified end-to-end — written on a machine with no GPU and no checkpoint
file to load; see SETUP.md for the real first run on the GPU host. YOLOv12
support depends on the installed `ultralytics` package version actually
supporting the v12 architecture — if loading fails, check the package version
against Ultralytics' YOLOv12 release notes.
"""

import io
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_THRESHOLD", "0.3"))
# Class-agnostic NMS threshold — Ultralytics' own internal NMS is per-class,
# so multiple different-class boxes stacked on the same real object (e.g.
# three boxes — Bird/Airplane/Drone — all on one actual bird) survive it.
# Collapse any pair of boxes overlapping more than this, regardless of
# predicted class, keeping only the higher-confidence one.
NMS_IOU_THRESHOLD = float(os.environ.get("YOLO_NMS_IOU_THRESHOLD", "0.35"))

CHECKPOINT_ENV = {
    "RGB": "YOLO_RGB_CHECKPOINT_PATH",
    "IR / THERMAL": "YOLO_IR_CHECKPOINT_PATH",
}

app = FastAPI()
_models: dict = {}


def _iou(box_a: list, box_b: list) -> float:
    """box_*: pixel [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _class_agnostic_nms(raw_detections: list, iou_threshold: float) -> list:
    """raw_detections: list of dicts each carrying a "_xyxy" pixel box (added
    by the caller, stripped before returning). Greedy NMS: highest confidence
    first, suppress anything that overlaps an already-kept box past the
    threshold — regardless of predicted class, since Ultralytics' own NMS is
    per-class and won't suppress a same-object, different-class duplicate."""
    ordered = sorted(raw_detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for det in ordered:
        if not any(_iou(det["_xyxy"], k["_xyxy"]) > iou_threshold for k in kept):
            kept.append(det)
    return kept


def _try_load(env_var: str):
    path = os.environ.get(env_var)
    if not path or not os.path.exists(path):
        return None
    from ultralytics import YOLO
    return YOLO(path)


@app.on_event("startup")
def load_models():
    for modality, env_var in CHECKPOINT_ENV.items():
        _models[modality] = _try_load(env_var)
    loaded = [m for m, v in _models.items() if v is not None]
    print(f"[yolo_server] loaded modalities: {loaded or 'NONE (check weights/ + env vars)'}")


@app.get("/health")
def health():
    return {"status": "ok", "loaded": [m for m, v in _models.items() if v is not None]}


@app.post("/detect")
async def detect(image: UploadFile = File(...), modality: str = Form(...)):
    model = _models.get(modality)
    if model is None:
        raise HTTPException(503, f"no checkpoint loaded for modality {modality!r}")

    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    w, h = img.size
    results = model.predict(img, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result = results[0]
    names = result.names  # {class_id: class_name}, embedded in the checkpoint

    detections = []
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = names.get(class_id, f"unknown_{class_id}")
            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                "_xyxy": [x1, y1, x2, y2],
            })

    kept = _class_agnostic_nms(detections, NMS_IOU_THRESHOLD)
    for det in kept:
        del det["_xyxy"]
        print(f"[DEBUG] kept class={det['class_name']} conf={det['confidence']:.2f} "
              f"bbox={det['bbox']}", flush=True)
    if len(kept) < len(detections):
        print(f"[DEBUG] NMS suppressed {len(detections) - len(kept)} overlapping "
              f"box(es) (threshold={NMS_IOU_THRESHOLD})", flush=True)

    return {"detections": kept}
