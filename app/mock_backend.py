"""
Placeholder inference layer.

Nothing here is a real model. run_rfdetr() and run_vlm() exist so the
Streamlit frontend can demonstrate the confidence-gated cascade end to end
before the trained RFDETR checkpoint and VLM API keys are wired in.

Swap points for the real system:
  - run_rfdetr(): replace body with a call into the trained RFDETR model
    (loaded from the lab checkpoint), keep the same return shape.
  - run_vlm(): replace body with an API call to the selected VLM provider,
    keep the same return shape.
"""

import hashlib
import random

CLASSES = ["Drone", "Bird", "Helicopter", "Airplane"]

_VLM_REASONS = {
    "Drone": "Rigid multi-rotor silhouette, stable hover geometry, sharp fixed-wing-absent profile consistent with a UAV airframe.",
    "Bird": "Flapping wing motion across frames, irregular flight path, no rigid rotor or fuselage structure. Consistent with avian flight.",
    "Helicopter": "Single large rotor disc visible, elongated tail boom, low-frequency blade blur pattern typical of rotary-wing aircraft.",
    "Airplane": "Fixed-wing planform, contrail/engine nacelle cues, high altitude cruise trajectory consistent with fixed-wing aircraft.",
}


def _seed_from(*parts) -> random.Random:
    key = "|".join(str(p) for p in parts).encode()
    digest = hashlib.sha256(key).hexdigest()
    return random.Random(int(digest[:16], 16))


def run_rfdetr(image_bytes: bytes, modality: str, model_name: str) -> dict:
    rng = _seed_from(hashlib.sha256(image_bytes).hexdigest(), modality, model_name)

    class_name = rng.choice(CLASSES)
    confidence = round(rng.uniform(0.20, 0.97), 2)

    box_w = rng.uniform(0.14, 0.32)
    box_h = rng.uniform(0.14, 0.32)
    box_x = rng.uniform(0.05, 0.95 - box_w)
    box_y = rng.uniform(0.05, 0.95 - box_h)

    return {
        "source": "RFDETR",
        "model_name": model_name,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": (box_x, box_y, box_w, box_h),  # fractional (x, y, w, h)
        "latency_ms": rng.randint(28, 65),
    }


def run_vlm(image_bytes: bytes, vlm_name: str) -> dict:
    rng = _seed_from(hashlib.sha256(image_bytes).hexdigest(), vlm_name, "vlm")

    class_name = rng.choice(CLASSES)
    confidence = round(rng.uniform(0.55, 0.96), 2)

    return {
        "source": "VLM",
        "model_name": vlm_name,
        "class_name": class_name,
        "confidence": confidence,
        "reasoning": _VLM_REASONS[class_name],
        "latency_ms": rng.randint(240, 480),
    }


def run_cascade(image_bytes: bytes, modality: str, rfdetr_model: str, vlm_model: str,
                 threshold: float, mode: str) -> dict:
    """
    mode: "RFDETR Only" | "VLM Only" | "RFDETR & VLM (Adaptive Fallback)"
    """
    result = {"rfdetr": None, "vlm": None, "cascaded": False, "final": None}

    if mode == "VLM Only":
        vlm_res = run_vlm(image_bytes, vlm_model)
        result["vlm"] = vlm_res
        result["final"] = vlm_res
        return result

    rfdetr_res = run_rfdetr(image_bytes, modality, rfdetr_model)
    result["rfdetr"] = rfdetr_res
    result["final"] = rfdetr_res

    if mode == "RFDETR Only":
        return result

    # Adaptive fallback
    if rfdetr_res["confidence"] < threshold:
        vlm_res = run_vlm(image_bytes, vlm_model)
        result["vlm"] = vlm_res
        result["cascaded"] = True
        result["final"] = vlm_res

    return result
