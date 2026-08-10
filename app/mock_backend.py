"""
Inference layer. Real inference (real_inference.py) is required by default —
if a model isn't configured or a real call fails, callers get a
ModelUnavailable error instead of a substituted result, since a guessed
class/bbox is worse than no answer at all. run_yolo()/run_vlm() (and their
multi-object counterparts) are the mock simulator; it only ever runs if
explicitly opted into via ALLOW_MOCK_FALLBACK=1 (useful for exercising the
rest of the app's logic on a machine with no GPU/models configured).
"""

import hashlib
import os
import random

import real_inference

ALLOW_MOCK_FALLBACK = os.environ.get("ALLOW_MOCK_FALLBACK", "").lower() in ("1", "true", "yes")

CLASSES = ["Airplane", "Bird", "Drone", "Helicopter"]


class ModelUnavailable(Exception):
    """Raised when real inference isn't configured or fails, and mock
    fallback is disabled (the default). Callers must surface this to the
    user rather than substitute a guessed result."""

CLASS_COLOR = {
    "Drone": "#B23A31",
    "Bird": "#9C6B0E",
    "Helicopter": "#6B4C9A",
    "Airplane": "#2C5A7C",
}

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


def run_yolo(image_bytes: bytes, modality: str, model_name: str) -> dict:
    rng = _seed_from(hashlib.sha256(image_bytes).hexdigest(), modality, model_name)

    class_name = rng.choice(CLASSES)
    confidence = round(rng.uniform(0.20, 0.97), 2)

    box_w = rng.uniform(0.14, 0.32)
    box_h = rng.uniform(0.14, 0.32)
    box_x = rng.uniform(0.05, 0.95 - box_w)
    box_y = rng.uniform(0.05, 0.95 - box_h)

    return {
        "source": "YOLO",
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


def _get_vlm_result(image_bytes: bytes, vlm_model: str) -> dict:
    if real_inference.vlm_available(vlm_model):
        try:
            return real_inference.run_real_vlm(image_bytes, vlm_model)
        except Exception as exc:
            if ALLOW_MOCK_FALLBACK:
                return run_vlm(image_bytes, vlm_model)
            raise ModelUnavailable(f"{vlm_model} failed: {exc}") from exc

    if ALLOW_MOCK_FALLBACK:
        return run_vlm(image_bytes, vlm_model)
    raise ModelUnavailable(f"{vlm_model} is not configured (no API key / service URL set).")


def _get_yolo_result(image_bytes: bytes, modality: str, rfdetr_model: str) -> tuple[dict, list]:
    """Returns (best_detection, all_detections). best_detection drives the
    existing single-verdict cascade/VLM-escalation logic unchanged;
    all_detections is every box YOLO actually found, for drawing every
    detected object rather than just the highest-confidence one."""
    if real_inference.yolo_available(modality):
        try:
            detections = real_inference.run_real_yolo(image_bytes, modality)
            if detections:
                return max(detections, key=lambda d: d["confidence"]), detections
            empty = {
                "source": "YOLO", "model_name": rfdetr_model, "class_name": "No Detection",
                "confidence": 0.0, "bbox": (0.0, 0.0, 1.0, 1.0), "latency_ms": 0,
            }
            return empty, []
        except Exception as exc:
            if ALLOW_MOCK_FALLBACK:
                mock_res = run_yolo(image_bytes, modality, rfdetr_model)
                return mock_res, [mock_res]
            raise ModelUnavailable(f"YOLO ({modality}) failed: {exc}") from exc

    if ALLOW_MOCK_FALLBACK:
        mock_res = run_yolo(image_bytes, modality, rfdetr_model)
        return mock_res, [mock_res]
    raise ModelUnavailable(f"YOLO ({modality}) is not configured (checkpoint/service missing).")


def run_cascade(image_bytes: bytes, modality: str, rfdetr_model: str, vlm_model: str,
                 threshold: float, mode: str) -> dict:
    """
    mode: "YOLO Only" | "VLM Only" | "YOLO & VLM (Adaptive Fallback)"
    """
    result = {"rfdetr": None, "rfdetr_all": [], "vlm": None, "cascaded": False, "final": None}

    if mode == "VLM Only":
        vlm_res = _get_vlm_result(image_bytes, vlm_model)
        result["vlm"] = vlm_res
        result["final"] = vlm_res
        return result

    rfdetr_res, rfdetr_all = _get_yolo_result(image_bytes, modality, rfdetr_model)
    result["rfdetr"] = rfdetr_res
    result["rfdetr_all"] = rfdetr_all
    result["final"] = rfdetr_res

    if mode == "YOLO Only":
        return result

    # Adaptive fallback
    if rfdetr_res["confidence"] < threshold:
        vlm_res = _get_vlm_result(image_bytes, vlm_model)
        result["vlm"] = vlm_res
        result["cascaded"] = True
        result["final"] = vlm_res

    return result


# ---------------------------------------------------------------------------
# Multi-object simulation for FOLDER / TEST FOLDER batch runs.
#
# Real YOLO/VLM inference produces zero or more detections per image. The
# single-detection functions above are kept as-is for the IMAGE tab; these
# functions simulate a believable multi-object detector instead, so batch.py
# can exercise the full matching/metrics/CSV/heatmap pipeline before the real
# model exists. Swap point for the real system: replace simulate_yolo_multi
# with real batched YOLO inference (returning the same detection dict shape,
# each optionally carrying which GTBox it was matched against isn't something
# a real model would know — that hint is dropped once real inference lands).
# ---------------------------------------------------------------------------

_MISS_PROB = 0.18          # probability YOLO fails to propose a box for a given GT object
_CORRECT_CLASS_PROB = 0.85  # probability a proposed box gets the right class
_STRAY_FP_PROB = 0.30       # probability of at least one spurious false-positive box per image
_VLM_CORRECT_PROB = 0.82    # probability the VLM fixes a detection to the true class


def _jitter_bbox(rng: random.Random, bbox) -> tuple:
    x, y, w, h = bbox
    dx = rng.uniform(-0.04, 0.04) * w
    dy = rng.uniform(-0.04, 0.04) * h
    dw = rng.uniform(-0.10, 0.10) * w
    dh = rng.uniform(-0.10, 0.10) * h

    nw = max(0.02, w + dw)
    nh = max(0.02, h + dh)
    nx = min(max(0.0, x + dx), 1.0 - nw)
    ny = min(max(0.0, y + dy), 1.0 - nh)
    return (nx, ny, nw, nh)


def simulate_yolo_multi(image_bytes: bytes, gt_boxes: list, modality: str, model_name: str) -> list:
    img_hash = hashlib.sha256(image_bytes).hexdigest()
    rng = _seed_from(img_hash, modality, model_name, "multi")

    detections = []

    for i, gt in enumerate(gt_boxes):
        if rng.random() < _MISS_PROB:
            continue

        if rng.random() < _CORRECT_CLASS_PROB:
            class_name = gt.class_name
            confidence = round(rng.uniform(0.55, 0.97), 2)
        else:
            class_name = rng.choice([c for c in CLASSES if c != gt.class_name])
            confidence = round(rng.uniform(0.15, 0.55), 2)

        detections.append({
            "source": "YOLO",
            "model_name": model_name,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": _jitter_bbox(rng, gt.bbox),
            "latency_ms": rng.randint(28, 65),
            "_gt": gt,
        })

    if rng.random() < _STRAY_FP_PROB:
        class_name = rng.choice(CLASSES)
        box_w = rng.uniform(0.08, 0.22)
        box_h = rng.uniform(0.08, 0.22)
        box_x = rng.uniform(0.0, 1.0 - box_w)
        box_y = rng.uniform(0.0, 1.0 - box_h)
        detections.append({
            "source": "YOLO",
            "model_name": model_name,
            "class_name": class_name,
            "confidence": round(rng.uniform(0.15, 0.55), 2),
            "bbox": (box_x, box_y, box_w, box_h),
            "latency_ms": rng.randint(28, 65),
            "_gt": None,
        })

    return detections


def simulate_vlm_for_detection(image_bytes: bytes, detection: dict, vlm_model: str,
                                true_class: str | None = None) -> dict:
    rng = _seed_from(hashlib.sha256(image_bytes).hexdigest(), vlm_model, "vlm-multi",
                      detection["bbox"], detection["class_name"])

    if true_class is not None:
        if rng.random() < _VLM_CORRECT_PROB:
            class_name = true_class
        else:
            class_name = rng.choice([c for c in CLASSES if c != true_class])
    else:
        class_name = rng.choice(CLASSES)

    confidence = round(rng.uniform(0.55, 0.96), 2)

    return {
        "source": "VLM",
        "model_name": vlm_model,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": detection["bbox"],
        "reasoning": _VLM_REASONS[class_name],
        "latency_ms": rng.randint(240, 480),
    }


def _get_yolo_multi_result(image_bytes: bytes, gt_boxes: list, modality: str, rfdetr_model: str) -> list:
    if real_inference.yolo_available(modality):
        try:
            detections = real_inference.run_real_yolo(image_bytes, modality)
            for d in detections:
                # Real detections carry no "which GT box was this simulated from"
                # hint — batch.py's VLM-scope metric falls back to a genuine
                # IoU lookup against gt_boxes when "_gt" is None (see
                # batch._find_gt_for_detection).
                d["_gt"] = None
            return detections
        except Exception as exc:
            if ALLOW_MOCK_FALLBACK:
                return simulate_yolo_multi(image_bytes, gt_boxes, modality, rfdetr_model)
            raise ModelUnavailable(f"YOLO ({modality}) failed: {exc}") from exc

    if ALLOW_MOCK_FALLBACK:
        return simulate_yolo_multi(image_bytes, gt_boxes, modality, rfdetr_model)
    raise ModelUnavailable(f"YOLO ({modality}) is not configured (checkpoint/service missing).")


def _get_vlm_multi_result(image_bytes: bytes, detection: dict, vlm_model: str,
                           true_class: str | None = None) -> dict:
    if real_inference.vlm_available(vlm_model):
        try:
            result = real_inference.run_real_vlm(image_bytes, vlm_model, crop_bbox=detection["bbox"])
            result["bbox"] = detection["bbox"]
            return result
        except Exception as exc:
            if ALLOW_MOCK_FALLBACK:
                return simulate_vlm_for_detection(image_bytes, detection, vlm_model, true_class=true_class)
            raise ModelUnavailable(f"{vlm_model} failed: {exc}") from exc

    if ALLOW_MOCK_FALLBACK:
        return simulate_vlm_for_detection(image_bytes, detection, vlm_model, true_class=true_class)
    raise ModelUnavailable(f"{vlm_model} is not configured (no API key / service URL set).")


def run_batch_cascade(image_bytes: bytes, gt_boxes: list, modality: str, rfdetr_model: str,
                       vlm_model: str, threshold: float, mode: str) -> dict:
    """
    Returns {"rfdetr": [...], "vlm": [...], "combined": [...]} — detection
    dicts (each may carry a private "_gt" key: the GTBox it was simulated
    from, or None for stray false positives / real detections). "combined" is
    what gets matched against ground truth for the pipeline's real end-to-end
    metrics and drawn on the annotated output image.
    """
    rfdetr_dets = _get_yolo_multi_result(image_bytes, gt_boxes, modality, rfdetr_model)

    if mode == "VLM Only":
        vlm_dets = []
        for d in rfdetr_dets:
            true_class = d["_gt"].class_name if d.get("_gt") else None
            v = _get_vlm_multi_result(image_bytes, d, vlm_model, true_class=true_class)
            v["_gt"] = d.get("_gt")
            v["cascaded"] = True
            vlm_dets.append(v)
        return {"rfdetr": [], "vlm": vlm_dets, "combined": vlm_dets}

    for d in rfdetr_dets:
        d["cascaded"] = False

    if mode == "YOLO Only":
        return {"rfdetr": rfdetr_dets, "vlm": [], "combined": rfdetr_dets}

    # Adaptive fallback: escalate anything under the confidence cutoff.
    vlm_dets = []
    combined = []
    for d in rfdetr_dets:
        if d["confidence"] < threshold:
            true_class = d["_gt"].class_name if d.get("_gt") else None
            v = _get_vlm_multi_result(image_bytes, d, vlm_model, true_class=true_class)
            v["_gt"] = d.get("_gt")
            v["cascaded"] = True
            v["rfdetr_confidence"] = d["confidence"]
            vlm_dets.append(v)
            combined.append(v)
        else:
            combined.append(d)

    return {"rfdetr": rfdetr_dets, "vlm": vlm_dets, "combined": combined}
