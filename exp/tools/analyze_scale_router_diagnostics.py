import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def finite_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def bbox_xywh_to_xyxy(bbox):
    if bbox is None or len(bbox) != 4:
        return None
    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


def bbox_iou_xywh(a, b):
    a = bbox_xywh_to_xyxy(a)
    b = bbox_xywh_to_xyxy(b)
    if a is None or b is None:
        return None
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return None
    return inter / union


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values, p):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * float(p) / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def numeric_stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": mean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
    }


def summarize_group(rows):
    selected = [row for row in rows if row["selected"]]
    source_counts = Counter(row["final_source"] for row in rows)
    selected_source_counts = Counter(row["final_source"] for row in selected)
    mlp_used = [row for row in rows if row["router_mlp_used"]]
    mlp_reject_reasons = Counter(row["router_mlp_reject_reason"] for row in rows if not row["router_mlp_used"])
    perceived_scale_groups = Counter(row["perceived_scale_group"] for row in rows if row["perceived_scale_group"] != "unknown")
    return {
        "count": len(rows),
        "selected": len(selected),
        "accepted_ratio": 0.0 if not rows else len(selected) / len(rows),
        "final_source": dict(sorted(source_counts.items())),
        "selected_final_source": dict(sorted(selected_source_counts.items())),
        "router_mlp_used": len(mlp_used),
        "router_mlp_used_ratio": 0.0 if not rows else len(mlp_used) / len(rows),
        "router_mlp_reject_reasons": dict(sorted(mlp_reject_reasons.items())),
        "router_mlp_perceived_scale_groups": dict(sorted(perceived_scale_groups.items())),
        "router_mlp_route_conf": numeric_stats([row["router_mlp_route_conf"] for row in rows]),
        "router_mlp_scale_conf": numeric_stats([row["router_mlp_scale_conf"] for row in rows]),
        "area_ratio": numeric_stats([row["area_ratio"] for row in rows]),
        "selected_area_ratio": numeric_stats([row["area_ratio"] for row in selected]),
        "center_shift": numeric_stats([row["center_shift"] for row in rows]),
        "selected_center_shift": numeric_stats([row["center_shift"] for row in selected]),
        "p2seg_sam_overlap": numeric_stats([row["p2seg_iou"] for row in rows]),
        "selected_p2seg_sam_overlap": numeric_stats([row["p2seg_iou"] for row in selected]),
        "same_class_point_hits": numeric_stats([row["same_class_point_hits"] for row in rows]),
        "other_class_point_hits": numeric_stats([row["other_class_point_hits"] for row in rows]),
        "gt_iou": numeric_stats([row["gt_iou"] for row in rows]),
        "selected_gt_iou": numeric_stats([row["gt_iou"] for row in selected]),
        "trust": numeric_stats([row["sam2_trust_score"] for row in rows]),
        "feature_consistency": numeric_stats([row["feature_consistency_score"] for row in rows]),
        "scale_consistency": numeric_stats([row["scale_consistency_score"] for row in rows]),
        "prompt_consensus_selected": sum(row["consensus_member_count"] > 1 for row in selected),
        "prompt_consensus_selected_ratio": 0.0 if not selected else sum(row["consensus_member_count"] > 1 for row in selected) / len(selected),
        "prompt_consensus_member_count": numeric_stats([row["consensus_member_count"] for row in rows if row["consensus_member_count"] > 0]),
        "prompt_consensus_weighted_iou": numeric_stats([row["consensus_weighted_iou"] for row in rows if row["consensus_member_count"] > 1]),
        "boundary_calibration_applied": sum(row["boundary_calibration_applied"] for row in rows),
        "boundary_calibration_applied_ratio": 0.0 if not rows else sum(row["boundary_calibration_applied"] for row in rows) / len(rows),
        "boundary_calibration_weight": numeric_stats([row["boundary_calibration_weight"] for row in rows if row["boundary_calibration_applied"]]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ScaleAwareRouter diagnostics without feeding GT back into training."
    )
    parser.add_argument("ann", help="Original COCO annotation JSON with GT boxes.")
    parser.add_argument("refined_result", help="Refined result JSON from refine_p2seg_with_sam2.py.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    args = parser.parse_args()

    dataset = load_json(args.ann)
    results = load_json(args.refined_result)
    anns_by_id = {ann["id"]: ann for ann in dataset.get("annotations", [])}

    rows = []
    missing_refine = 0
    missing_ann = 0
    gate_reasons = Counter()
    mlp = Counter()
    mlp_reject_reasons = Counter()
    mlp_perceived_scale_groups = Counter()
    ambiguities = Counter()
    scale_head_reasons = Counter()
    for pred in results:
        ann_id = pred.get("ann_id")
        ann = anns_by_id.get(ann_id)
        if ann is None:
            missing_ann += 1
            continue
        info = pred.get("sam2_refine")
        if not isinstance(info, dict):
            missing_refine += 1
            continue

        final_bbox = info.get("final_bbox") if info.get("selected", False) else pred.get("bbox")
        # true_bbox is train-only diagnostic metadata; it must never be written
        # into pseudo annotations or used by the production router.
        gt_iou = bbox_iou_xywh(ann.get("true_bbox"), final_bbox)
        selected = bool(info.get("selected", False))
        gate_reason = info.get("gate_reason", "unknown")
        gate_reasons[gate_reason] += 1
        ambiguities[info.get("scale_ambiguity", "unknown")] += 1
        scale_head_reasons[info.get("scale_head_reason", "unknown")] += 1
        router_mlp_used = bool(info.get("router_mlp_used", False))
        mlp["used" if router_mlp_used else "not_used"] += 1
        if not router_mlp_used:
            mlp_reject_reasons[info.get("router_mlp_reject_reason", "") or "unknown"] += 1
        perceived_scale_group = info.get("perceived_scale_group", "unknown")
        if perceived_scale_group != "unknown":
            mlp_perceived_scale_groups[perceived_scale_group] += 1
        rows.append({
            "scale_group": info.get("scale_group", "unknown"),
            "scale_abs_group": info.get("scale_abs_group", "unknown"),
            "scale_relative_group": info.get("scale_relative_group", "unknown"),
            "category_id": int(ann.get("category_id", -1)),
            "scale_ambiguity": info.get("scale_ambiguity", "unknown"),
            "scale_head_reason": info.get("scale_head_reason", "unknown"),
            "final_source": info.get("final_source", "unknown"),
            "gate_reason": gate_reason,
            "selected": selected,
            "router_mlp_used": router_mlp_used,
            "router_mlp_reject_reason": info.get("router_mlp_reject_reason", "") or "unknown",
            "router_mlp_route_conf": finite_float(info.get("router_mlp_route_conf")),
            "router_mlp_scale_conf": finite_float(info.get("router_mlp_scale_conf")),
            "perceived_scale_group": perceived_scale_group,
            "area_ratio": finite_float(info.get("area_ratio")),
            "center_shift": finite_float(info.get("center_shift")),
            "p2seg_iou": finite_float(info.get("p2seg_iou")),
            "same_class_point_hits": finite_float(info.get("same_class_point_hits")),
            "other_class_point_hits": finite_float(info.get("other_class_point_hits")),
            "gt_iou": gt_iou,
            "sam2_trust_score": finite_float(info.get("sam2_trust_score")),
            "feature_consistency_score": finite_float(info.get("feature_consistency_score")),
            "scale_consistency_score": finite_float(info.get("scale_consistency_score")),
            "consensus_member_count": int(info.get("consensus_member_count", 0) or 0),
            "consensus_weighted_iou": finite_float(info.get("consensus_weighted_iou")),
            "boundary_calibration_applied": bool(info.get("boundary_calibration_applied", False)),
            "boundary_calibration_weight": finite_float(info.get("boundary_calibration_weight")),
        })

    by_scale = defaultdict(list)
    by_abs_scale = defaultdict(list)
    by_relative_scale = defaultdict(list)
    by_category = defaultdict(list)
    by_source = defaultdict(list)
    by_gate_reason = defaultdict(list)
    for row in rows:
        by_scale[row["scale_group"]].append(row)
        by_abs_scale[row["scale_abs_group"]].append(row)
        by_relative_scale[row["scale_relative_group"]].append(row)
        by_category[row["category_id"]].append(row)
        by_source[row["final_source"]].append(row)
        by_gate_reason[row.get("gate_reason", "unknown")].append(row)

    summary = {
        "count": len(rows),
        "missing_ann": missing_ann,
        "missing_refine": missing_refine,
        "overall": summarize_group(rows),
        "by_scale_group": {k: summarize_group(v) for k, v in sorted(by_scale.items())},
        "by_scale_abs_group": {k: summarize_group(v) for k, v in sorted(by_abs_scale.items())},
        "by_scale_relative_group": {k: summarize_group(v) for k, v in sorted(by_relative_scale.items())},
        "by_category": {str(k): summarize_group(v) for k, v in sorted(by_category.items())},
        "by_final_source": {k: summarize_group(v) for k, v in sorted(by_source.items())},
        "by_gate_reason": {k: summarize_group(v) for k, v in sorted(by_gate_reason.items())},
        "gate_reasons": dict(sorted(gate_reasons.items())),
        "scale_ambiguities": dict(sorted(ambiguities.items())),
        "scale_head_reasons": dict(sorted(scale_head_reasons.items())),
        "router_mlp": {
            **dict(sorted(mlp.items())),
            "used_ratio": 0.0 if not rows else mlp.get("used", 0) / len(rows),
        },
        "router_mlp_reject_reasons": dict(sorted(mlp_reject_reasons.items())),
        "router_mlp_perceived_scale_groups": dict(sorted(mlp_perceived_scale_groups.items())),
    }

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.out:
        dump_json(summary, args.out)


if __name__ == "__main__":
    main()
