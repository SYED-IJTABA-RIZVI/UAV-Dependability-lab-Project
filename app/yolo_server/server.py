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

CHECKPOINT_ENV = {
    "RGB": "YOLO_RGB_CHECKPOINT_PATH",
    "IR / THERMAL": "YOLO_IR_CHECKPOINT_PATH",
}

app = FastAPI()
_models: dict = {}


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
            print(f"[DEBUG] class={class_name} conf={confidence:.2f} "
                  f"raw_xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})", flush=True)
            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
            })

    return {"detections": detections}
