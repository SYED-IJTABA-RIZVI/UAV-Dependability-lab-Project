"""One-shot diagnostic script (not the FastAPI server): runs the trained
RFDETR checkpoint over a YOLO-format test folder (images/ + labels/) and
dumps BOTH the raw model output and the ground-truth boxes to one CSV, in
the same fractional top-left (x, y, w, h) coordinate space server.py
converts predictions into. That makes it possible to line up a prediction
against its matching ground truth row and see exactly how far off (and in
which direction) the box is — offset, scale, or axis-swap — without needing
to touch/trust the rest of the app pipeline.

Test folder layout expected at TEST_FOLDER (default /test_data):
    images/            <image files>
    labels/            <one .txt per image, same stem, YOLO format>
    classes.txt        <optional, one class name per line>

Run via the "rfdetr-debug" service in docker-compose.yml (profile "debug").
"""

import csv
import os
from pathlib import Path

from PIL import Image

CLASSES = ["Airplane", "Bird", "Drone", "Helicopter"]  # must match server.py
DEFAULT_CLASSES = CLASSES
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

TEST_FOLDER = os.environ.get("TEST_FOLDER", "/test_data")
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "/output/debug_predictions.csv")
MODALITY = os.environ.get("MODALITY", "RGB")
CONFIDENCE_THRESHOLD = float(os.environ.get("RFDETR_THRESHOLD", "0.3"))

CHECKPOINT_ENV = {
    "RGB": "RFDETR_RGB_CHECKPOINT_PATH",
    "IR / THERMAL": "RFDETR_IR_CHECKPOINT_PATH",
}


def load_class_map(root: Path) -> list[str]:
    classes_txt = root / "classes.txt"
    if classes_txt.exists():
        names = [line.strip() for line in classes_txt.read_text().splitlines() if line.strip()]
        if names:
            return names
    return DEFAULT_CLASSES


def parse_label_file(path: Path, class_map: list[str]) -> list[tuple[str, tuple]]:
    """Returns [(class_name, (x, y, w, h))], top-left origin, fractional 0-1."""
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id, cx, cy, w, h = parts
        class_id = int(class_id)
        if class_id < 0 or class_id >= len(class_map):
            continue
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        boxes.append((class_map[class_id], (cx - w / 2, cy - h / 2, w, h)))
    return boxes


def main() -> None:
    root = Path(TEST_FOLDER)
    images_dir = root / "images"
    labels_dir = root / "labels"
    class_map = load_class_map(root)

    checkpoint_path = os.environ.get(CHECKPOINT_ENV[MODALITY])
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise SystemExit(f"No checkpoint found at {checkpoint_path!r} for modality {MODALITY!r}")

    from rfdetr import RFDETRBase
    model = RFDETRBase(pretrained_weights=checkpoint_path)

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["type", "filename", "img_w", "img_h", "class_name", "confidence",
                  "raw_x1", "raw_y1", "raw_x2", "raw_y2", "norm_x", "norm_y", "norm_w", "norm_h"]

    n_images, n_preds, n_gts = 0, 0, 0
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            n_images += 1

            img = Image.open(image_path).convert("RGB")
            w, h = img.size

            result = model.predict(img, threshold=CONFIDENCE_THRESHOLD)
            for xyxy, confidence, class_id in zip(result.xyxy, result.confidence, result.class_id):
                class_id = int(class_id)
                class_name = CLASSES[class_id] if 0 <= class_id < len(CLASSES) else f"unknown_{class_id}"
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                writer.writerow({
                    "type": "pred", "filename": image_path.name, "img_w": w, "img_h": h,
                    "class_name": class_name, "confidence": float(confidence),
                    "raw_x1": x1, "raw_y1": y1, "raw_x2": x2, "raw_y2": y2,
                    "norm_x": x1 / w, "norm_y": y1 / h,
                    "norm_w": (x2 - x1) / w, "norm_h": (y2 - y1) / h,
                })
                n_preds += 1

            label_path = labels_dir / f"{image_path.stem}.txt"
            for class_name, (gx, gy, gw, gh) in parse_label_file(label_path, class_map):
                writer.writerow({
                    "type": "gt", "filename": image_path.name, "img_w": w, "img_h": h,
                    "class_name": class_name, "confidence": "",
                    "raw_x1": "", "raw_y1": "", "raw_x2": "", "raw_y2": "",
                    "norm_x": gx, "norm_y": gy, "norm_w": gw, "norm_h": gh,
                })
                n_gts += 1

    print(f"[debug_predict] {n_images} images, {n_preds} predictions, {n_gts} ground-truth boxes")
    print(f"[debug_predict] wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
