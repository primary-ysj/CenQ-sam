import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refine_p2seg_with_sam2 import (
    ROUTER_FEATURE_KEYS,
    ROUTER_FEATURE_KEYS_BY_VERSION,
    ROUTER_ROUTE_LABELS,
    ROUTER_SCALE_LABELS,
)

FEATURE_KEYS = ROUTER_FEATURE_KEYS
ROUTE_LABELS = ROUTER_ROUTE_LABELS
SCALE_LABELS = ROUTER_SCALE_LABELS


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(obj, f)


def finite_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if value != value or value in (float("inf"), float("-inf")):
        return default
    return value


def high_confidence_teacher(info, min_trust, min_margin, min_feature, min_scale):
    if not info.get("selected", False):
        return False
    if finite_float(info.get("sam2_trust_score")) < min_trust:
        return False
    if finite_float(info.get("score_margin")) < min_margin:
        return False
    if finite_float(info.get("scale_consistency_score")) < min_scale:
        return False
    if "feature_consistency_score" in info and finite_float(info.get("feature_consistency_score")) < min_feature:
        return False
    if int(info.get("same_class_point_hits", 0)) > 0:
        return False
    return True


def route_p2seg_teacher(info):
    return (
        not info.get("selected", False)
        and info.get("gate_reason") == "route_p2seg"
        and info.get("final_source") == "p2seg"
    )


def teacher_labels(info):
    scale_group = info.get("scale_group", "unknown")
    safety_group = info.get("scale_safety_group", info.get("scale_abs_group", scale_group))
    final_source = info.get("final_source", "p2seg")
    if safety_group == "small" and final_source == "sam2":
        final_source = "fusion"
    elif scale_group == "medium" and final_source == "sam2":
        final_source = "fusion"
    elif scale_group != "large" and final_source == "sam2":
        final_source = "fusion"
    return scale_group, final_source


def main():
    parser = argparse.ArgumentParser(
        description="Export ScaleAwareRouter diagnostics as pseudo-labeled MLP training data."
    )
    parser.add_argument("refined_result", help="SAM2 refined result JSON produced by refine_p2seg_with_sam2.py.")
    parser.add_argument("out", help="Output JSONL file.")
    parser.add_argument("--router-head-version", type=int, choices=sorted(ROUTER_FEATURE_KEYS_BY_VERSION.keys()), default=2)
    parser.add_argument("--min-trust", type=float, default=0.70)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--min-feature", type=float, default=0.30)
    parser.add_argument("--min-scale", type=float, default=0.45)
    args = parser.parse_args()

    feature_keys = ROUTER_FEATURE_KEYS_BY_VERSION[int(args.router_head_version)]
    results = load_json(args.refined_result)
    rows = []
    skipped = 0
    for pred in results:
        info = pred.get("sam2_refine")
        if not isinstance(info, dict):
            skipped += 1
            continue
        is_p2seg_teacher = route_p2seg_teacher(info)
        if not is_p2seg_teacher and not high_confidence_teacher(info, args.min_trust, args.min_margin, args.min_feature, args.min_scale):
            skipped += 1
            continue
        scale_group, final_source = teacher_labels(info)
        if info.get("scale_safety_group", info.get("scale_abs_group")) == "small" and final_source == "sam2":
            skipped += 1
            continue
        if scale_group not in SCALE_LABELS or final_source not in ROUTE_LABELS:
            skipped += 1
            continue
        rows.append({
            "image_id": pred.get("image_id"),
            "ann_id": pred.get("ann_id"),
            "category_id": pred.get("category_id"),
            "features": [finite_float(info.get(key)) for key in feature_keys],
            "feature_keys": feature_keys,
            "router_head_version": int(args.router_head_version),
            "scale_label": SCALE_LABELS[scale_group],
            "route_label": ROUTE_LABELS[final_source],
            "fusion_weight": finite_float(info.get("fusion_weight")),
            "release_score": 1.0 if final_source in {"fusion", "sam2", "feature"} else 0.0,
            "fusion_weight_delta": max(0.0, finite_float(info.get("fusion_weight")) - 0.5),
            "teacher_type": "route_p2seg" if is_p2seg_teacher else "high_confidence",
        })

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"exported={len(rows)}, skipped={skipped}, output={args.out}")


if __name__ == "__main__":
    main()
