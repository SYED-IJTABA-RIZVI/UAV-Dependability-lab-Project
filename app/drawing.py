"""Shared PIL drawing helpers for annotated output images.
Used by main.py (IMAGE tab, single detection) and batch.py (FOLDER).
"""

from PIL import Image, ImageDraw


def _tag(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, color: str, text_color: str = "white") -> None:
    text_w = max(len(text) * 6 + 10, 40)
    ty = max(y - 16, 0)
    draw.rectangle([x, ty, x + text_w, ty + 14], fill=color)
    draw.text((x + 4, ty + 1), text, fill=text_color)


def draw_single(image: Image.Image, bbox, label: str, color: str) -> Image.Image:
    img = image.convert("RGB").copy()
    w, h = img.size
    x, y, bw, bh = bbox
    px, py, pw, ph = x * w, y * h, bw * w, bh * h

    draw = ImageDraw.Draw(img)
    draw.rectangle([px, py, px + pw, py + ph], outline=color, width=3)
    _tag(draw, px, py, label, color)
    return img


def draw_multi(image: Image.Image, detections: list, class_color: dict, source_label: str = "YOLO") -> Image.Image:
    """One box + label per detection. Used so every detected object gets
    drawn, not just the single highest-confidence one."""
    img = image.convert("RGB").copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)

    for det in detections:
        x, y, bw, bh = det["bbox"]
        px, py, pw, ph = x * w, y * h, bw * w, bh * h
        color = class_color.get(det["class_name"], "#2C5A7C")
        draw.rectangle([px, py, px + pw, py + ph], outline=color, width=3)
        label = f"{det['class_name']} · {det['confidence']:.2f} · {source_label}"
        _tag(draw, px, py, label, color)

    return img
