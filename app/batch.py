"""Batch orchestration for FOLDER (no ground truth) runs. Writes annotated
images + a CSV to the user-provided output path, and persists everything to
Postgres via db.py.

Predicted classes are always drawn from mock_backend.CLASSES (the fixed
Airplane/Bird/Drone/Helicopter set the model — real or mock — predicts).
"""

import csv
import io
from pathlib import Path

from PIL import Image

import db
from drawing import draw_multi
from labels import IMAGE_EXTS
from mock_backend import CLASS_COLOR, run_cascade


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
        annotated = draw_multi(img, result.get("rfdetr_all", []), CLASS_COLOR)
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


