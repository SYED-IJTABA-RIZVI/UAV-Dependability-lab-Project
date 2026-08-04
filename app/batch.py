"""Batch orchestration for FOLDER (no ground truth) and TEST FOLDER (YOLO-labeled
eval) runs. Writes annotated images + CSVs + confusion-matrix/IoU heatmaps to the
user-provided output path, and persists everything to Postgres via db.py.

Predicted classes are always drawn from mock_backend.CLASSES (the fixed
Drone/Bird/Helicopter/Airplane set the model — real or mock — predicts). A
custom classes.txt/data.yaml in a test folder should use the same class names;
mismatched names are skipped when building confusion matrices rather than
crashing the run.
"""

import csv
import hashlib
import io
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import db
from drawing import draw_multi_eval, draw_single
from labels import IMAGE_EXTS, discover_test_folder, load_class_map, parse_label_file
from metrics import compute_scope_metrics, iou, undetected_class_breakdown
from mock_backend import CLASS_COLOR, run_batch_cascade, run_cascade


def _seed_pass_time(image_bytes: bytes, tag: str) -> random.Random:
    digest = hashlib.sha256(image_bytes + tag.encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _safe_idx(class_map: list, name: str):
    return class_map.index(name) if name in class_map else None


def _find_gt_for_detection(det: dict, gt_boxes: list, iou_threshold: float):
    """Which GT box (if any) this detection's bbox actually localizes.
    Mock detections carry a "_gt" hint (the exact GTBox they were jittered
    from); real detections don't, so we fall back to a genuine IoU search —
    same threshold/logic real end-to-end matching uses."""
    if det.get("_gt") is not None:
        return det["_gt"]
    best_gt, best_iou = None, 0.0
    for gt in gt_boxes:
        score = iou(det["bbox"], gt.bbox)
        if score > best_iou:
            best_gt, best_iou = gt, score
    return best_gt if best_iou >= iou_threshold else None


def run_folder(folder_path: str, output_path: str, modality: str, rfdetr_model: str,
               vlm_model: str, threshold: float, mode: str) -> dict:
    input_dir = Path(folder_path)
    out_dir = Path(output_path)
    annotated_dir = out_dir / "annotated_images"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    fieldnames = ["filename", "class_name", "confidence", "source", "cascaded", "vlm_response",
                  "bbox_x", "bbox_y", "bbox_w", "bbox_h", "latency_ms"]
    csv_rows = []
    per_image_results = []
    images_ui = []
    num_cascaded = 0

    for p in image_paths:
        image_bytes = p.read_bytes()
        result = run_cascade(image_bytes, modality, rfdetr_model, vlm_model, threshold, mode)
        img = Image.open(io.BytesIO(image_bytes))

        final = result["final"]
        label = f"{final['class_name']} · {final['confidence']:.2f} · {final['source']}"
        annotated = draw_single(img, final["bbox"], label, CLASS_COLOR.get(final["class_name"], "#2C5A7C"))
        annotated_path = annotated_dir / p.name
        annotated.convert("RGB").save(annotated_path)

        if result["cascaded"]:
            num_cascaded += 1

        csv_rows.append({
            "filename": p.name,
            "class_name": final["class_name"],
            "confidence": final["confidence"],
            "source": final["source"],
            "cascaded": result["cascaded"],
            "vlm_response": result["vlm"]["reasoning"] if result.get("vlm") else "",
            "bbox_x": final["bbox"][0], "bbox_y": final["bbox"][1],
            "bbox_w": final["bbox"][2], "bbox_h": final["bbox"][3],
            "latency_ms": final["latency_ms"],
        })
        per_image_results.append({"filename": p.name, "result": result, "annotated_path": str(annotated_path)})
        images_ui.append({"filename": p.name, "annotated_path": str(annotated_path)})

    results_csv = out_dir / "results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    db.save_folder_run(folder_path, output_path, modality, rfdetr_model, vlm_model, threshold, mode,
                        per_image_results)

    return {
        "num_images": len(image_paths),
        "num_cascaded": num_cascaded,
        "output_path": str(out_dir),
        "results_csv": str(results_csv),
        "images": images_ui,
    }


def _finalize_scope(agg: dict) -> dict:
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    mean_iou = agg["iou_sum"] / agg["iou_n"] if agg["iou_n"] > 0 else 0.0
    avg_time = agg["time_sum"] / agg["time_n"] if agg["time_n"] > 0 else 0.0
    images_per_sec = 1000.0 / avg_time if avg_time > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        "mean_iou": mean_iou, "avg_time_per_image_ms": avg_time, "images_per_sec": images_per_sec,
    }


def _save_confusion_heatmap(matrix: np.ndarray, class_map: list, scope: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_map)))
    ax.set_xticklabels(class_map, rotation=45, ha="right")
    ax.set_yticks(range(len(class_map)))
    ax.set_yticklabels(class_map)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"{scope.upper()} confusion matrix")

    max_val = int(matrix.max()) if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if max_val > 0 and matrix[i, j] > max_val / 2 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_iou_histogram(iou_values: list, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    if iou_values:
        ax.hist(iou_values, bins=20, range=(0, 1), color="#2C5A7C")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.set_title("IoU distribution (matched detections)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_test_folder(test_folder_path: str, output_path: str, modality: str, rfdetr_model: str,
                     vlm_model: str, threshold: float, mode: str, iou_threshold: float = 0.5) -> dict:
    class_map = load_class_map(test_folder_path)
    pairs = discover_test_folder(test_folder_path)

    out_dir = Path(output_path)
    annotated_dir = out_dir / "annotated_images"
    heatmaps_dir = out_dir / "heatmaps"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    image_summary_rows = []
    per_image_for_db = []
    images_ui = []
    all_iou_values = []
    undetected_fn_boxes = []

    agg = {scope: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "iou_n": 0, "time_sum": 0.0, "time_n": 0}
           for scope in ("rfdetr", "vlm", "combined")}
    confusion_totals = {scope: np.zeros((len(class_map), len(class_map)), dtype=int)
                         for scope in ("rfdetr", "vlm", "combined")}

    for image_path, label_path in pairs:
        image_bytes = image_path.read_bytes()
        gt_boxes = parse_label_file(label_path, class_map) if label_path else []

        batch_result = run_batch_cascade(image_bytes, gt_boxes, modality, rfdetr_model, vlm_model,
                                          threshold, mode)
        rfdetr_dets, vlm_dets, combined_dets = (
            batch_result["rfdetr"], batch_result["vlm"], batch_result["combined"])

        rfdetr_pass_ms = _seed_pass_time(image_bytes, "rfdetr_pass").randint(28, 65)
        vlm_total_ms = sum(d["latency_ms"] for d in vlm_dets)
        if mode == "VLM Only":
            time_ms = vlm_total_ms
        elif mode == "RFDETR Only":
            time_ms = rfdetr_pass_ms
        else:
            time_ms = rfdetr_pass_ms + vlm_total_ms

        # RFDETR-scope metrics (the model alone, pre-cascade)
        if mode != "VLM Only":
            r = compute_scope_metrics(rfdetr_dets, gt_boxes, iou_threshold)
            agg["rfdetr"]["tp"] += r["tp"]; agg["rfdetr"]["fp"] += r["fp"]; agg["rfdetr"]["fn"] += r["fn"]
            for _, _, iv in r["match"].tp:
                agg["rfdetr"]["iou_sum"] += iv; agg["rfdetr"]["iou_n"] += 1
            agg["rfdetr"]["time_sum"] += rfdetr_pass_ms; agg["rfdetr"]["time_n"] += 1
            for pred, gt, _ in r["match"].localized_pairs:
                gi, pi = _safe_idx(class_map, gt.class_name), _safe_idx(class_map, pred["class_name"])
                if gi is not None and pi is not None:
                    confusion_totals["rfdetr"][gi, pi] += 1

        # VLM-scope metrics: correction accuracy on exactly the cascaded subset
        # (VLM never re-localizes, so this is class-match on the RFDETR-proposed
        # box — via the mock's "_gt" hint when present, else a real IoU lookup)
        for d in vlm_dets:
            gt = _find_gt_for_detection(d, gt_boxes, iou_threshold)
            if gt is not None:
                iv = iou(d["bbox"], gt.bbox)
                gi = _safe_idx(class_map, gt.class_name)
                if d["class_name"] == gt.class_name:
                    agg["vlm"]["tp"] += 1
                    agg["vlm"]["iou_sum"] += iv; agg["vlm"]["iou_n"] += 1
                else:
                    agg["vlm"]["fp"] += 1
                    agg["vlm"]["fn"] += 1
                pi = _safe_idx(class_map, d["class_name"])
                if gi is not None and pi is not None:
                    confusion_totals["vlm"][gi, pi] += 1
            else:
                agg["vlm"]["fp"] += 1
        agg["vlm"]["time_sum"] += vlm_total_ms; agg["vlm"]["time_n"] += 1

        # Combined scope: the real end-to-end pipeline result
        c = compute_scope_metrics(combined_dets, gt_boxes, iou_threshold)
        agg["combined"]["tp"] += c["tp"]; agg["combined"]["fp"] += c["fp"]; agg["combined"]["fn"] += c["fn"]
        for _, _, iv in c["match"].tp:
            agg["combined"]["iou_sum"] += iv; agg["combined"]["iou_n"] += 1
            all_iou_values.append(iv)
        agg["combined"]["time_sum"] += time_ms; agg["combined"]["time_n"] += 1
        for pred, gt, _ in c["match"].localized_pairs:
            gi, pi = _safe_idx(class_map, gt.class_name), _safe_idx(class_map, pred["class_name"])
            if gi is not None and pi is not None:
                confusion_totals["combined"][gi, pi] += 1

        fn_gt_ids = {id(gt) for gt in c["match"].fn}
        undetected_fn_boxes.extend(c["match"].fn)

        img = Image.open(io.BytesIO(image_bytes))
        annotated = draw_multi_eval(img, gt_boxes, combined_dets, fn_gt_ids, CLASS_COLOR)
        annotated_path = annotated_dir / image_path.name
        annotated.convert("RGB").save(annotated_path)
        images_ui.append({"filename": image_path.name, "annotated_path": str(annotated_path)})

        matched_gt_for_pred = {id(pred): gt for pred, gt, _ in c["match"].localized_pairs}
        for det in combined_dets:
            gt = matched_gt_for_pred.get(id(det))
            is_tp = gt is not None and det["class_name"] == gt.class_name
            detail_rows.append({
                "filename": image_path.name,
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "source": det["source"],
                "cascaded": det.get("cascaded", False),
                "vlm_response": det.get("reasoning", ""),
                "bbox_x": det["bbox"][0], "bbox_y": det["bbox"][1],
                "bbox_w": det["bbox"][2], "bbox_h": det["bbox"][3],
                "matched_gt_class": gt.class_name if gt else "",
                "iou": round(iou(det["bbox"], gt.bbox), 3) if gt else "",
                "match_result": "TP" if is_tp else "FP",
                "latency_ms": det["latency_ms"],
            })

        image_summary_rows.append({
            "filename": image_path.name,
            "num_gt": len(gt_boxes),
            "num_pred": len(combined_dets),
            "num_cascaded": sum(1 for d in combined_dets if d.get("cascaded")),
            "time_ms": time_ms,
        })

        per_image_for_db.append({
            "filename": image_path.name,
            "gt_boxes": gt_boxes,
            "combined": combined_dets,
            "combined_match": c["match"],
            "time_ms": time_ms,
            "annotated_path": str(annotated_path),
        })

    scope_metrics = {scope: _finalize_scope(agg[scope]) for scope in ("rfdetr", "vlm", "combined")}
    undetected_breakdown = undetected_class_breakdown(undetected_fn_boxes)
    undetected_counts: dict = {}
    for gt in undetected_fn_boxes:
        undetected_counts[gt.class_name] = undetected_counts.get(gt.class_name, 0) + 1

    results_csv = out_dir / "results.csv"
    with open(results_csv, "w", newline="") as f:
        fieldnames = ["filename", "class_name", "confidence", "source", "cascaded", "vlm_response",
                      "bbox_x", "bbox_y", "bbox_w", "bbox_h", "matched_gt_class", "iou",
                      "match_result", "latency_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    images_summary_csv = out_dir / "images_summary.csv"
    with open(images_summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "num_gt", "num_pred", "num_cascaded", "time_ms"])
        writer.writeheader()
        writer.writerows(image_summary_rows)

    metrics_csv = out_dir / "metrics_summary.csv"
    with open(metrics_csv, "w", newline="") as f:
        fieldnames = ["scope", "precision", "recall", "f1", "accuracy", "mean_iou",
                      "avg_time_per_image_ms", "images_per_sec", "tp", "fp", "fn"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scope, m in scope_metrics.items():
            writer.writerow({"scope": scope, **m})

    undetected_csv = out_dir / "undetected_by_class.csv"
    with open(undetected_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_name", "percentage"])
        writer.writeheader()
        for cls, pct in undetected_breakdown.items():
            writer.writerow({"class_name": cls, "percentage": pct})

    heatmap_paths = {}
    for scope in ("rfdetr", "vlm", "combined"):
        path = heatmaps_dir / f"confusion_{scope}.png"
        _save_confusion_heatmap(confusion_totals[scope], class_map, scope, path)
        heatmap_paths[scope] = str(path)

    iou_path = heatmaps_dir / "iou_distribution.png"
    _save_iou_histogram(all_iou_values, iou_path)
    heatmap_paths["iou_distribution"] = str(iou_path)

    db.save_eval_run(test_folder_path, output_path, modality, rfdetr_model, vlm_model, threshold, mode,
                      iou_threshold, per_image_for_db, scope_metrics, undetected_breakdown, undetected_counts)

    return {
        "num_images": len(pairs),
        "scope_metrics": scope_metrics,
        "undetected_breakdown": undetected_breakdown,
        "undetected_counts": undetected_counts,
        "output_path": str(out_dir),
        "csv_paths": {
            "results": str(results_csv),
            "images_summary": str(images_summary_csv),
            "metrics_summary": str(metrics_csv),
            "undetected_by_class": str(undetected_csv),
        },
        "heatmap_paths": heatmap_paths,
        "images": images_ui,
    }
