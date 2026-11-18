"""Flag disagreements between YOLO labels and cached checker predictions."""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")
    settings = cfg["second_opinion"]
    min_iou = settings["iou_missing"]
    issues = []
    for image in ds.image_files():
        stem = image.stem
        if stem not in preds:
            continue
        lines = ds.read_label_lines(stem)
        size = ds.image_size(stem)
        if lines is None or size is None:
            continue

        labels = []
        for line, raw in lines:
            try:
                cls, *box = parse_yolo_line(raw)
            except (ValueError, OverflowError):
                continue
            if 0 <= cls < len(ds.classes):
                labels.append((line, cls, box))

        for cls, conf, *box in preds[stem]:
            if conf < settings["conf"] or box[2] * box[3] * size[0] * size[1] < settings["min_area_px"]:
                continue
            candidates = [(label_cls, iou_xywh(box, label_box))
                          for _, label_cls, label_box in labels]
            candidates = [(label_cls, iou) for label_cls, iou in candidates if iou >= min_iou]
            if not candidates:
                code = "POSSIBLE_MISSING"
                best_iou = 0.0
            else:
                same = [iou for label_cls, iou in candidates if label_cls == cls]
                if same:
                    best_iou = max(same)
                    if best_iou >= settings["iou_loose"]:
                        continue
                    code = "POSSIBLE_LOOSE_BOX"
                else:
                    best_iou = max(iou for _, iou in candidates)
                    if best_iou < settings["iou_match"]:
                        continue
                    code = "POSSIBLE_WRONG_CLASS"
            issues.append(Issue("second_opinion", WARN, code, stem, "Dự đoán và nhãn cần đối chiếu",
                                extra={"conf": conf, "iou": best_iou}))

        support = [pred[2:] for pred in preds[stem] if pred[1] >= settings["fp_conf"]]
        for line, _, box in labels:
            if all(iou_xywh(box, pred) < min_iou for pred in support):
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_FALSE_POS", stem,
                                    "Box nhãn thiếu dự đoán tương ứng", line))
    return issues
