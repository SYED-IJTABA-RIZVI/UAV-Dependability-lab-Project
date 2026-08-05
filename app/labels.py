"""YOLO-format label parsing for the TEST FOLDER evaluation flow.

Expects a test folder shaped like:
    test_folder/
        images/  <image files>
        labels/  <one .txt per image, same stem>

Label lines are `class_id cx cy w h`, all normalized 0-1 — the same
fractional (x, y, w, h) convention already used for bboxes in mock_backend.py,
except YOLO's cx/cy are box centers, so they get converted to top-left here.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CLASSES = ["Airplane", "Bird", "Drone", "Helicopter"]  # lab's real training order (alphabetical)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class GTBox:
    class_name: str
    bbox: tuple  # fractional (x, y, w, h), top-left origin


def load_class_map(test_folder: str) -> list[str]:
    root = Path(test_folder)

    classes_txt = root / "classes.txt"
    if classes_txt.exists():
        names = [line.strip() for line in classes_txt.read_text().splitlines() if line.strip()]
        if names:
            return names

    data_yaml = root / "data.yaml"
    if data_yaml.exists():
        data = yaml.safe_load(data_yaml.read_text()) or {}
        names = data.get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names, key=int)]
        if isinstance(names, list) and names:
            return names

    return DEFAULT_CLASSES


def parse_label_file(path: Path, class_map: list[str]) -> list[GTBox]:
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
        x, y = cx - w / 2, cy - h / 2
        boxes.append(GTBox(class_name=class_map[class_id], bbox=(x, y, w, h)))

    return boxes


def discover_test_folder(root: str) -> list[tuple[Path, Path | None]]:
    root = Path(root)
    images_dir = root / "images"
    labels_dir = root / "labels"

    pairs = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        pairs.append((image_path, label_path if label_path.exists() else None))

    return pairs
