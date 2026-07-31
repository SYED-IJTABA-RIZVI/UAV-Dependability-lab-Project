"""Pure detection-evaluation math: IoU, matching, precision/recall/F1/accuracy,
undetected-class breakdown, confusion matrices. No I/O here — app/batch.py
handles reading images/labels and writing CSVs/plots.

Predictions are dicts with at least: class_name, confidence, bbox (fractional
x, y, w, h). Ground truths are labels.GTBox instances.
"""

from dataclasses import dataclass, field

import numpy as np


def iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class MatchResult:
    tp: list = field(default_factory=list)   # (prediction, gt, iou) — localized AND class matches
    fp: list = field(default_factory=list)   # predictions never localized, or localized w/ wrong class
    fn: list = field(default_factory=list)   # gt never localized, or localized w/ wrong class
    localized_pairs: list = field(default_factory=list)  # (prediction, gt, iou) regardless of class — for confusion matrix


def match_detections(predictions, ground_truths, iou_threshold: float = 0.5) -> MatchResult:
    result = MatchResult()

    unmatched_gt = list(ground_truths)
    ordered_preds = sorted(predictions, key=lambda p: p["confidence"], reverse=True)

    for pred in ordered_preds:
        best_gt, best_iou = None, 0.0
        for gt in unmatched_gt:
            score = iou(pred["bbox"], gt.bbox)
            if score > best_iou:
                best_gt, best_iou = gt, score

        if best_gt is not None and best_iou >= iou_threshold:
            unmatched_gt.remove(best_gt)
            result.localized_pairs.append((pred, best_gt, best_iou))
            if pred["class_name"] == best_gt.class_name:
                result.tp.append((pred, best_gt, best_iou))
            else:
                result.fp.append(pred)
                result.fn.append(best_gt)
        else:
            result.fp.append(pred)

    result.fn.extend(unmatched_gt)
    return result


def compute_scope_metrics(predictions, ground_truths, iou_threshold: float = 0.5) -> dict:
    match = match_detections(predictions, ground_truths, iou_threshold)

    tp, fp, fn = len(match.tp), len(match.fp), len(match.fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    mean_iou = float(np.mean([m[2] for m in match.tp])) if match.tp else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": accuracy, "mean_iou": mean_iou,
        "match": match,
    }


def undetected_class_breakdown(fn_boxes) -> dict:
    if not fn_boxes:
        return {}

    counts: dict[str, int] = {}
    for gt in fn_boxes:
        counts[gt.class_name] = counts.get(gt.class_name, 0) + 1

    total = sum(counts.values())
    return {cls: round(100.0 * n / total, 1) for cls, n in counts.items()}


def confusion_matrix(predictions, ground_truths, classes: list, iou_threshold: float = 0.5) -> np.ndarray:
    match = match_detections(predictions, ground_truths, iou_threshold)
    idx = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for pred, gt, _ in match.localized_pairs:
        if gt.class_name in idx and pred["class_name"] in idx:
            matrix[idx[gt.class_name], idx[pred["class_name"]]] += 1

    return matrix
