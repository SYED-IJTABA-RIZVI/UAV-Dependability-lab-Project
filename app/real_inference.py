"""
Real-inference dispatch layer — the "swap points" mock_backend.py's docstring
promised. Every function here either succeeds with a real result or raises;
callers in mock_backend.py catch and fall back to the simulator, so a missing
checkpoint / unreachable local VLM service / OpenRouter timeout degrades
gracefully instead of crashing the app.

YOLO (YOLOv12 for RGB, YOLOv10 for IR/Thermal) runs as its own GPU-enabled
container (app/yolo_server/), and the 3 self-hosted VLMs (InternVL2.5,
DeepSeek-VL, BLIP-2) share a SINGLE GPU-enabled container (app/vlm_server/)
that lazy-loads whichever one was actually requested and unloads it when a
different one is requested — deliberate VRAM tradeoff: only one local VLM
occupies VRAM at a time (plus YOLO, always resident), at the cost of a
reload-latency hit on the first call after switching models. See
docker-compose.yml's "gpu" profile. The main `app` service stays
GPU-reservation-free so `docker compose up` keeps working on machines with no
GPU. Qwen2.5-VL is the one VLM with a real hosted API (OpenRouter) and is
called directly from here.

NOTE: YOLO and the 3 local VLM servers cannot be exercised on a machine
without a GPU. This module is written to each library's documented API but
the first real end-to-end test is on the GPU host — see SETUP.md.
"""

import base64
import json
import os
import time

import requests

QWEN_MODEL_ID = "qwen/qwen2.5-vl-72b-instruct"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

REQUEST_TIMEOUT_S = 20
# Local VLM calls can trigger an unload-current/load-new-model swap
# (app/vlm_server/server.py) before inference even starts — an 8B model's
# checkpoint shards alone took ~25s+ to load in testing. 20s would false-fail
# on every model switch, not just genuinely stuck requests.
LOCAL_VLM_TIMEOUT_S = 180

CLASSES = ["Airplane", "Bird", "Drone", "Helicopter"]
_CLASS_LOOKUP = {c.lower(): c for c in CLASSES}


def _normalize_class(raw_name: str, source: str) -> str:
    """Real models (YOLO checkpoints, VLM text responses) aren't guaranteed to
    return this app's exact casing for a class name — a checkpoint's embedded
    labels might be "drone" where this app expects "Drone", for example.
    Match case-insensitively (and tolerate stray whitespace) rather than
    rejecting an otherwise-correct result over casing alone."""
    canonical = _CLASS_LOOKUP.get(str(raw_name).strip().lower())
    if canonical is None:
        raise ValueError(f"{source} returned unknown class: {raw_name!r}")
    return canonical


VLM_MODEL_IDS = {
    "InternVL2.5": "OpenGVLab/InternVL2_5-8B",
    "DeepSeek-VL": "deepseek-ai/deepseek-vl-7b-chat",
    "BLIP-2": "Salesforce/blip2-opt-2.7b",
}

YOLO_CHECKPOINT_ENV = {
    "RGB": "YOLO_RGB_CHECKPOINT_PATH",
    "IR / THERMAL": "YOLO_IR_CHECKPOINT_PATH",
}


def _classification_prompt() -> str:
    return (
        "You are classifying an aerial sky object into exactly one of these "
        f"classes: {', '.join(CLASSES)}. "
        "Respond with ONLY a JSON object, no markdown fences, no extra text: "
        '{"class_name": "<one of the classes above>", '
        '"confidence": <float 0-1>, "reasoning": "<one sentence>"}'
    )


def _parse_vlm_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text)
    class_name = _normalize_class(data["class_name"], "VLM")
    return {
        "class_name": class_name,
        "confidence": float(data["confidence"]),
        "reasoning": str(data.get("reasoning", "")),
    }


def vlm_available(vlm_name: str) -> bool:
    if vlm_name == "Qwen2.5-VL":
        return bool(os.environ.get("OPENROUTER_API_KEY"))
    return vlm_name in VLM_MODEL_IDS and bool(os.environ.get("VLM_SERVICE_URL"))


def _crop(image_bytes: bytes, crop_bbox) -> bytes:
    """crop_bbox: fractional (x, y, w, h). Returns re-encoded JPEG bytes."""
    if crop_bbox is None:
        return image_bytes
    from io import BytesIO

    from PIL import Image
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    x, y, bw, bh = crop_bbox
    box = (int(x * w), int(y * h), int((x + bw) * w), int((y + bh) * h))
    cropped = img.crop(box)
    buf = BytesIO()
    cropped.save(buf, format="JPEG")
    return buf.getvalue()


def _call_qwen_openrouter(image_bytes: bytes) -> dict:
    api_key = os.environ["OPENROUTER_API_KEY"]
    b64 = base64.b64encode(image_bytes).decode()

    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": QWEN_MODEL_ID,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": _classification_prompt()},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return _parse_vlm_json(text)


def _call_local_vlm(image_bytes: bytes, vlm_name: str) -> dict:
    """Calls the single shared vlm_server, telling it which model_id to run —
    the server lazy-loads/swaps models on demand (see docstring above), so
    this can block for a while on the first call after switching VLMs."""
    base_url = os.environ["VLM_SERVICE_URL"]
    resp = requests.post(
        f"{base_url}/classify",
        files={"image": ("image.jpg", image_bytes, "image/jpeg")},
        data={"prompt": _classification_prompt(), "model_id": VLM_MODEL_IDS[vlm_name]},
        timeout=LOCAL_VLM_TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    class_name = _normalize_class(data["class_name"], "VLM")
    return {
        "class_name": class_name,
        "confidence": float(data["confidence"]),
        "reasoning": str(data.get("reasoning", "")),
    }


def run_real_vlm(image_bytes: bytes, vlm_name: str, crop_bbox=None) -> dict:
    """
    Raises on any failure — caller falls back to the mock simulator.
    crop_bbox (fractional x, y, w, h): if given, crop to that region before
    sending — used in batch/eval mode so the VLM judges the specific detected
    object rather than the whole scene. None (IMAGE tab) sends the full image.
    """
    image_bytes = _crop(image_bytes, crop_bbox)

    start = time.monotonic()
    if vlm_name == "Qwen2.5-VL":
        result = _call_qwen_openrouter(image_bytes)
    else:
        result = _call_local_vlm(image_bytes, vlm_name)
    latency_ms = int((time.monotonic() - start) * 1000)

    return {
        "source": "VLM",
        "model_name": vlm_name,
        "class_name": result["class_name"],
        "confidence": result["confidence"],
        "reasoning": result["reasoning"] or "(no reasoning returned by model)",
        "latency_ms": latency_ms,
    }


def yolo_available(modality: str) -> bool:
    env_name = YOLO_CHECKPOINT_ENV.get(modality)
    if not env_name:
        return False
    return bool(os.environ.get("YOLO_SERVICE_URL")) and bool(os.environ.get(env_name))


def run_real_yolo(image_bytes: bytes, modality: str) -> list:
    """
    Calls the yolo service (app/yolo_server/), which loads the checkpoint
    named by YOLO_RGB_CHECKPOINT_PATH / YOLO_IR_CHECKPOINT_PATH (YOLOv12 for
    RGB, YOLOv10 for IR/Thermal).
    Raises on any failure — caller falls back to the mock simulator.
    Returns a list of detection dicts, same shape as mock_backend's.
    """
    base_url = os.environ["YOLO_SERVICE_URL"]
    start = time.monotonic()
    resp = requests.post(
        f"{base_url}/detect",
        files={"image": ("image.jpg", image_bytes, "image/jpeg")},
        data={"modality": modality},
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    latency_ms = int((time.monotonic() - start) * 1000)
    detections = resp.json()["detections"]

    for det in detections:
        det["class_name"] = _normalize_class(det["class_name"], "YOLO")
        det["source"] = "YOLO"
        det["model_name"] = f"YOLO-{modality} (real)"
        det["bbox"] = tuple(det["bbox"])
        det.setdefault("latency_ms", latency_ms)

    return detections
