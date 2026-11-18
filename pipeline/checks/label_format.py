"""Check the syntax and geometry of YOLO labels."""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line


def run(ds, cfg):
    settings = cfg["label_format"]
    issues = []
    for image in ds.image_files():
        seen = []
        size = ds.image_size(image.stem)
        for line, raw in ds.read_label_lines(image.stem) or []:
            try:
                cls, x, y, w, h = parse_yolo_line(raw)
            except (ValueError, OverflowError):
                issues.append(Issue("label_format", ERROR, "BAD_FORMAT", image.stem, "Dòng YOLO không hợp lệ", line))
                continue
            if not 0 <= cls < len(ds.classes):
                issues.append(Issue("label_format", ERROR, "BAD_CLASS_ID", image.stem, "ID lớp ngoài danh sách", line))
                continue
            if w <= 0 or h <= 0:
                issues.append(Issue("label_format", ERROR, "BOX_ZERO", image.stem, "Kích thước box không dương", line))
                continue

            box = (x, y, w, h)
            overflow = max(0, w / 2 - x, h / 2 - y, x + w / 2 - 1, y + h / 2 - 1)
            if overflow > settings["overflow_tol"] + 1e-12:
                issues.append(Issue("label_format", ERROR, "BOX_OUT_OF_IMAGE", image.stem,
                                    "Box tràn mép ảnh", line, {"overflow": overflow}))
            if any(cls == prior_cls and iou_xywh(box, prior) > settings["dup_box_iou"]
                   for prior_cls, prior in seen):
                issues.append(Issue("label_format", ERROR, "DUP_BOX", image.stem, "Box trùng cùng lớp", line))
            seen.append((cls, box))

            if size and min(w * size[0], h * size[1]) < settings["min_box_px"]:
                issues.append(Issue("label_format", WARN, "BOX_TINY", image.stem, "Cạnh box quá nhỏ", line))
            if w * h > settings["max_box_frac"]:
                issues.append(Issue("label_format", WARN, "BOX_HUGE", image.stem, "Diện tích box quá lớn", line))
    return issues
