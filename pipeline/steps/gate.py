"""Decide whether a reviewed batch can be released."""
from ..common import ERROR, WARN


def decide(issues, image_stems, cfg, reviewed=frozenset()):
    stems = set(image_stems)
    n_errors = 0
    warn_images = set()
    for issue in issues:
        if issue.severity == ERROR:
            n_errors += 1
        elif issue.severity == WARN and issue.image in stems and issue.image not in reviewed:
            warn_images.add(issue.image)

    n_images = len(stems)
    ratio = len(warn_images) / n_images if n_images else 0.0
    limit = cfg["gate"]["warn_ratio_hold"]
    if n_errors:
        decision = "Rework"
        reason = f"Còn {n_errors} lỗi bắt buộc."
    elif ratio >= limit:
        decision = "Hold"
        reason = f"Cảnh báo trên {len(warn_images)}/{n_images} ảnh chưa duyệt, đạt ngưỡng {limit:.0%}."
    else:
        decision = "Release"
        reason = f"Hết lỗi bắt buộc, {len(warn_images)}/{n_images} ảnh còn cảnh báo chưa duyệt."
    return dict(decision=decision, n_errors=n_errors, n_warn_images=len(warn_images),
                n_images=n_images, warn_ratio=round(ratio, 4), reasons=[reason])
