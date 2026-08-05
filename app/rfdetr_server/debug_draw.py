"""One-shot script: run the real checkpoint on a single image, print the
model's raw xyxy for every detection, and draw those exact boxes onto the
image so they can be eyeballed against the real object location.

Draws directly from the raw pixel xyxy (no normalize/denormalize round
trip) — mathematically identical to what server.py's fractional bbox would
produce once drawing.py multiplies it back out by the same image's w/h, so
this is a faithful preview of what the app draws, with the raw numbers
printed alongside it for comparison.

Usage (inside the rfdetr container, which already has the checkpoint):
    docker compose exec rfdetr python debug_draw.py /path/to/image.jpg /path/to/output.jpg [MODALITY]

MODALITY defaults to "RGB". Use "IR / THERMAL" for the IR checkpoint.
"""

import os
import sys

from PIL import Image, ImageDraw

CLASSES = ["Airplane", "Bird", "Drone", "Helicopter"]  # must match server.py
CHECKPOINT_ENV = {
    "RGB": "RFDETR_RGB_CHECKPOINT_PATH",
    "IR / THERMAL": "RFDETR_IR_CHECKPOINT_PATH",
}


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: python debug_draw.py <input_image> <output_image> [MODALITY]")

    input_path, output_path = sys.argv[1], sys.argv[2]
    modality = sys.argv[3] if len(sys.argv) > 3 else "RGB"

    checkpoint_path = os.environ.get(CHECKPOINT_ENV[modality])
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise SystemExit(f"No checkpoint found at {checkpoint_path!r} for modality {modality!r}")

    from rfdetr import RFDETRBase
    model = RFDETRBase(pretrained_weights=checkpoint_path)

    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    print(f"image: {input_path}  size: w={w} h={h}")

    threshold = float(os.environ.get("RFDETR_THRESHOLD", "0.3"))
    result = model.predict(img, threshold=threshold)

    draw = ImageDraw.Draw(img)
    n = 0
    for xyxy, confidence, class_id in zip(result.xyxy, result.confidence, result.class_id):
        class_id = int(class_id)
        class_name = CLASSES[class_id] if 0 <= class_id < len(CLASSES) else f"unknown_{class_id}"
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        print(f"  raw_xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})  "
              f"class={class_name}  confidence={float(confidence):.2f}")

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        label = f"{class_name} {float(confidence):.2f}"
        draw.rectangle([x1, max(y1 - 16, 0), x1 + len(label) * 7 + 8, max(y1 - 16, 0) + 14], fill="red")
        draw.text((x1 + 4, max(y1 - 16, 0) + 1), label, fill="white")
        n += 1

    if n == 0:
        print("  no detections above threshold")

    img.save(output_path)
    print(f"saved annotated image to {output_path}")


if __name__ == "__main__":
    main()
