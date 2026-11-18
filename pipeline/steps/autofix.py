"""Apply deterministic repairs to the working dataset and record them."""
from collections import defaultdict

import yaml

from ..common import ERROR, parse_yolo_line


DROP = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}


def clipped_label(raw):
    cls, x, y, w, h = parse_yolo_line(raw)
    left, right = max(0, x - w / 2), min(1, x + w / 2)
    top, bottom = max(0, y - h / 2), min(1, y + h / 2)
    if right <= left or bottom <= top:
        return ""
    values = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
    return f"{cls} " + " ".join(repr(value) for value in values)


def fix(ds, issues, cfg):
    errors = [issue for issue in issues if issue.severity == ERROR]
    removed = set()
    changes = []
    by_image = defaultdict(dict)
    pending_labels = set()

    def record(image, line, action, before, after):
        changes.append(dict(image=image, line=line, action=action, before=before, after=after))

    for issue in errors:
        if issue.code not in REMOVE and not issue.code.startswith("DUP_IMAGE_"):
            continue
        stem = issue.image
        if stem in removed:
            continue
        removed.add(stem)
        image = ds.image_path(stem)
        label = ds.label_path(stem)
        if image is not None or label.exists():
            before = image.name if image is not None else label.name
            if image is not None:
                image.unlink()
            label.unlink(missing_ok=True)
            record(stem, None, "remove_image", before, "")

    for issue in errors:
        stem = issue.image
        if stem in removed:
            continue
        if issue.code in DROP:
            by_image[stem][issue.line] = "drop_line"
        elif issue.code == "BOX_OUT_OF_IMAGE":
            by_image[stem].setdefault(issue.line, "clip_box")
        elif issue.code == "MISSING_LABEL_FILE" and not ds.label_path(stem).exists():
            if stem not in pending_labels:
                pending_labels.add(stem)
                record(stem, None, "needs_label", "(không có file)", "chuyển người gán")
        elif issue.code == "BAD_CLASS_NAME":
            config = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) if ds.yaml_path.exists() else {}
            config = config or {}
            before = config.get("names")
            if before != cfg["classes"]:
                config["names"] = list(cfg["classes"])
                ds.yaml_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
                record("data.yaml", None, "rename_classes", str(before), str(config["names"]))

    for stem, actions in by_image.items():
        path = ds.label_path(stem)
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8", errors="replace").splitlines()
        updated = []
        for line, raw in enumerate(original, 1):
            action = actions.get(line)
            if action is None:
                updated.append(raw)
                continue
            after = clipped_label(raw) if action == "clip_box" else ""
            if after:
                updated.append(after)
            record(stem, line, "clip_box" if after else "drop_line", raw, after)
        if updated != original:
            path.write_text("\n".join(updated) + ("\n" if updated else ""), encoding="utf-8")
    return changes
