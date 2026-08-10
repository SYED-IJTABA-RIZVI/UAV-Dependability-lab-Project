"""Shared PIL drawing helpers for annotated output images.
Used by main.py (IMAGE tab, single detection) and batch.py (FOLDER / TEST FOLDER).
"""

import math

from PIL import Image, ImageDraw


def _tag(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, color: str, text_color: str = "white") -> None:
    text_w = max(len(text) * 6 + 10, 40)
    ty = max(y - 16, 0)
    draw.rectangle([x, ty, x + text_w, ty + 14], fill=color)
    draw.text((x + 4, ty + 1), text, fill=text_color)


def _dashed_rectangle(draw: ImageDraw.ImageDraw, box, color: str, width: int = 2,
                       dash_len: int = 6, gap_len: int = 4) -> None:
    x0, y0, x1, y1 = box

    def dashed_line(p0, p1):
        lx0, ly0 = p0
        lx1, ly1 = p1
        length = math.hypot(lx1 - lx0, ly1 - ly0)
        if length == 0:
            return
        dx, dy = (lx1 - lx0) / length, (ly1 - ly0) / length
        dist, draw_seg = 0.0, True
        while dist < length:
            seg_len = dash_len if draw_seg else gap_len
            seg_end = min(dist + seg_len, length)
            if draw_seg:
                draw.line([(lx0 + dx * dist, ly0 + dy * dist), (lx0 + dx * seg_end, ly0 + dy * seg_end)],
                          fill=color, width=width)
            dist = seg_end
            draw_seg = not draw_seg

    dashed_line((x0, y0), (x1, y0))
    dashed_line((x1, y0), (x1, y1))
    dashed_line((x1, y1), (x0, y1))
    dashed_line((x0, y1), (x0, y0))


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
    """One box + label per detection, no ground-truth overlay (that's
    draw_multi_eval, for TEST FOLDER). Used so every detected object gets
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


def draw_multi_eval(image: Image.Image, gt_boxes: list, combined_dets: list, fn_gt_ids: set,
                     class_color: dict) -> Image.Image:
    """
    gt_boxes: list of labels.GTBox
    combined_dets: list of detection dicts (class_name, confidence, bbox, source)
    fn_gt_ids: set of id(gt) for GT boxes never matched by the combined pipeline (undetected)
    """
    img = image.convert("RGB").copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Matched GT (thin gray, untagged) and predictions first, so an UNDETECTED
    # tag on a box a wrong-class prediction happens to sit on top of doesn't
    # get drawn over and hidden — undetected GT is drawn last, on top.
    for gt in gt_boxes:
        if id(gt) not in fn_gt_ids:
            x, y, bw, bh = gt.bbox
            px, py, pw, ph = x * w, y * h, bw * w, bh * h
            _dashed_rectangle(draw, (px, py, px + pw, py + ph), "#8D959D", width=1)

    for det in combined_dets:
        x, y, bw, bh = det["bbox"]
        px, py, pw, ph = x * w, y * h, bw * w, bh * h
        color = class_color.get(det["class_name"], "#2C5A7C")
        draw.rectangle([px, py, px + pw, py + ph], outline=color, width=3)
        label = f"{det['class_name']} · {det['confidence']:.2f} · {det['source']}"
        _tag(draw, px, py, label, color)

    for gt in gt_boxes:
        if id(gt) in fn_gt_ids:
            x, y, bw, bh = gt.bbox
            px, py, pw, ph = x * w, y * h, bw * w, bh * h
            _dashed_rectangle(draw, (px, py, px + pw, py + ph), "#B23A31", width=2)
            _tag(draw, px, py + ph + 4, f"{gt.class_name.upper()} · UNDETECTED", "#B23A31")

    return img
