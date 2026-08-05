"""
FastAPI wrapper around Roboflow's `rfdetr` package, serving the lab's two
trained checkpoints (RGB and IR/Thermal). NOT verified end-to-end — written
on a machine with no GPU and no checkpoint file to load; see SETUP.md for the
real first run on the GPU host.

Assumes the checkpoints were trained with class ids in the same order used
everywhere else in this app (labels.py, mock_backend.CLASSES):
0=Drone, 1=Bird, 2=Helicopter, 3=Airplane. If the lab's training config used
a different order, fix CLASSES below to match.
"""

import io
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

CLASSES = ["Drone", "Bird", "Helicopter", "Airplane"]
CONFIDENCE_THRESHOLD = float(os.environ.get("RFDETR_THRESHOLD", "0.3"))

CHECKPOINT_ENV = {
    "RGB": "RFDETR_RGB_CHECKPOINT_PATH",
    "IR / THERMAL": "RFDETR_IR_CHECKPOINT_PATH",
}

app = FastAPI()
_models: dict = {}


def _try_load(env_var: str):
    path = os.environ.get(env_var)
    if not path or not os.path.exists(path):
        return None
    from rfdetr import RFDETRBase
    return RFDETRBase(pretrained_weights=path)


@app.on_event("startup")
def load_models():
    for modality, env_var in CHECKPOINT_ENV.items():
        _models[modality] = _try_load(env_var)
    loaded = [m for m, v in _models.items() if v is not None]
    print(f"[rfdetr_server] loaded modalities: {loaded or 'NONE (check weights/ + env vars)'}")


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
    result = model.predict(img, threshold=CONFIDENCE_THRESHOLD)
    print(f"[DEBUG] input image size: w={w} h={h}", flush=True)

    detections = []
    for xyxy, confidence, class_id in zip(result.xyxy, result.confidence, result.class_id):
        class_id = int(class_id)
        if class_id < 0 or class_id >= len(CLASSES):
            continue
        x1, y1, x2, y2 = xyxy
        bbox = [float(x1) / w, float(y1) / h, float(x2 - x1) / w, float(y2 - y1) / h]
        print(f"[DEBUG] class={CLASSES[class_id]} conf={float(confidence):.2f} "
              f"raw_xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) -> norm_bbox={bbox}", flush=True)
        detections.append({
            "class_name": CLASSES[class_id],
            "confidence": float(confidence),
            "bbox": bbox,
        })

    return {"detections": detections}
