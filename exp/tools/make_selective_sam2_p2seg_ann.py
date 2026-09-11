import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from pycocotools import mask as maskUtils


def progress_iter(iterable, total=None, desc=None, disable=False):
    if disable:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(obj, f)


def segm_to_rle(segm, height, width):
    if isinstance(segm, list):
        return maskUtils.merge(maskUtils.frPyObjects(segm, height, width))
    if isinstance(segm, dict):
        if isinstance(segm.get("counts"), list):
            return maskUtils.frPyObjects(segm, height, width)
        return segm
    raise TypeError(f"Unsupported segmentation type: {type(segm)}")


def rle_to_jsonable(rle):
    out = dict(rle)
    if isinstance(out.get("counts"), bytes):
        out["counts"] = out["counts"].decode("ascii")
    return out


def sanitize_xywh_bbox(bbox, width, height):
    if bbox is None or len(bbox) != 4:
        return None
    x, y, w, h = [float(v) for v in bbox]
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    if w <= 0 or h <= 0:
        return None
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def bbox_from_result(pred, rle):
    refine_info = pred.get("sam2_refine")
    if isinstance(refine_info, dict) and refine_info.get("selected", False):
        final_bbox = refine_info.get("final_bbox")
        if isinstance(final_bbox, list) and len(final_bbox) == 4:
            return [float(v) for v in final_bbox], "router_final_bbox"
    if "bbox" in pred:
        return [float(v) for v in pred["bbox"]], "pred_bbox"
    return [float(v) for v in maskUtils.toBbox(rle).tolist()], "mask_bbox"


def ann_weight_from_pred(pred, mode):
    if mode == "none":
        return None
    refine_info = pred.get("sam2_refine")
    score = float(pred.get("score", 1.0))
    if mode == "score" or not isinstance(refine_info, dict):
        return score
    if mode == "trust":
        return float(refine_info.get("sam2_trust_score", score))
    if mode == "hybrid":
        trust = float(refine_info.get("sam2_trust_score", score))
        scale = float(refine_info.get("scale_consistency_score", 1.0))
        feature = float(refine_info.get("feature_consistency_score", 1.0))
        return max(0.0, min(1.0, 0.4 * score + 0.3 * trust + 0.15 * scale + 0.15 * feature))
    raise ValueError(f"Unsupported ann_weight_mode: {mode}")


def compact_refine_info(refine_info):
    if not isinstance(refine_info, dict):
        return None
    keys = [
        "selected",
        "gate_reason",
        "final_source",
        "scale_group",
        "scale_prob",
        "sam2_trust_score",
        "feature_consistency_score",
        "scale_consistency_score",
        "prompt_consistency_score",
        "fusion_weight",
        "area_ratio",
        "p2seg_iou",
        "center_shift",
        "feature_box_iou",
        "feature_support_ratio",
        "local_point_density",
        "same_class_point_hits",
        "other_class_point_hits",
        "scale_abs_group",
        "scale_relative_group",
        "scale_candidate_abs_group",
        "scale_ambiguity",
        "scale_head_reason",
        "router_mlp_used",
        "router_mlp_route_conf",
        "router_mlp_scale_conf",
        "router_mlp_scale_group",
        "router_mlp_reject_reason",
        "route_profile",
        "route_scale_level",
        "route_support_score",
        "route_effective_trust",
        "route_dynamic_fusion_cap",
    ]
    return {key: refine_info[key] for key in keys if key in refine_info}


def accepted_scale(scale_group, accept_scales):
    return scale_group in accept_scales


def should_use_sam2(refined_pred, args):
    refine_info = refined_pred.get("sam2_refine")
    if not isinstance(refine_info, dict):
        return False, "missing_refine"
    if not refine_info.get("selected", False):
        return False, f"not_selected:{refine_info.get('gate_reason', 'unknown')}"
    scale_group = refine_info.get("scale_group", "unknown")
    if not accepted_scale(scale_group, args.accept_sam2_scales):
        return False, f"blocked_scale:{scale_group}"
    if float(refine_info.get("sam2_trust_score", 0.0)) < args.min_trust:
        return False, "low_trust"
    if float(refine_info.get("feature_consistency_score", 0.0)) < args.min_feature:
        return False, "low_feature"
    if float(refine_info.get("scale_consistency_score", 0.0)) < args.min_scale:
        return False, "low_scale"
    if float(refine_info.get("p2seg_iou", 0.0)) < args.min_p2seg_iou:
        return False, "low_p2seg_iou"
    if int(refine_info.get("same_class_point_hits", 0)) > args.max_same_class_hits:
        return False, "same_class_hits"
    return True, "sam2_selected"


def update_ann_from_pred(ann, pred, img_info, source, bbox_source, ann_weight_mode):
    rle = segm_to_rle(pred["segmentation"], img_info["height"], img_info["width"])
    bbox, bbox_source_value = bbox_from_result(pred, rle)
    if bbox_source == "mask":
        bbox = [float(v) for v in maskUtils.toBbox(rle).tolist()]
        bbox_source_value = "mask_bbox"
    clean_bbox = sanitize_xywh_bbox(bbox, img_info["width"], img_info["height"])
    if clean_bbox is None:
        return False, "invalid_bbox"
    area = float(maskUtils.area(rle))
    if area <= 0:
        area = float(clean_bbox[2] * clean_bbox[3])
    if area <= 0:
        return False, "empty_mask"

    ann["bbox"] = clean_bbox
    ann["area"] = float(area)
    ann["segmentation"] = rle_to_jsonable(rle)
    ann["selective_sam2_source"] = source
    ann["selective_sam2_bbox_source"] = bbox_source_value
    ann_weight = ann_weight_from_pred(pred, ann_weight_mode)
    if ann_weight is None:
        ann.pop("ann_weight", None)
    else:
        ann["ann_weight"] = float(ann_weight)
    return True, "updated"


def validate_pred(ann, pred):
    if ann["image_id"] != pred.get("image_id"):
        raise AssertionError(f"image_id mismatch for ann_id={ann['id']}")
    if ann["category_id"] != pred.get("category_id"):
        raise AssertionError(f"category_id mismatch for ann_id={ann['id']}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a selective SAM2/P2Seg pseudo-mask annotation for second-stage P2Seg training."
    )
    parser.add_argument("ori_ann", help="Original COCO-format point annotation JSON.")
    parser.add_argument("base_p2seg_result", help="Original P2Seg mask result JSON with ann_id.")
    parser.add_argument("sam2_refined_result", help="SAM2 refined result JSON with sam2_refine diagnostics.")
    parser.add_argument("save_ann", help="Output COCO annotation JSON for second-stage P2Seg training.")
    parser.add_argument(
        "--accept-sam2-scales",
        nargs="+",
        default=["medium", "large"],
        choices=["small", "medium", "large"],
        help="Scale groups allowed to use selected SAM2 masks. Default: medium large.",
    )
    parser.add_argument("--min-trust", type=float, default=0.0)
    parser.add_argument("--min-feature", type=float, default=0.0)
    parser.add_argument("--min-scale", type=float, default=0.0)
    parser.add_argument("--min-p2seg-iou", type=float, default=0.0)
    parser.add_argument("--max-same-class-hits", type=int, default=0)
    parser.add_argument(
        "--sam2-bbox-source",
        choices=["router", "mask"],
        default="mask",
        help="Use router final_bbox when available or always derive bbox from mask.",
    )
    parser.add_argument(
        "--ann-weight-mode",
        choices=["none", "score", "trust", "hybrid"],
        default="none",
        help="Optional annotation weights copied from the chosen prediction.",
    )
    parser.add_argument(
        "--write-refine-diagnostics",
        action="store_true",
        help="Copy compact router diagnostics to selected output annotations.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    dataset = load_json(args.ori_ann)
    base_results = load_json(args.base_p2seg_result)
    refined_results = load_json(args.sam2_refined_result)

    imgs_by_id = {img["id"]: img for img in dataset["images"]}
    base_by_ann_id = {pred.get("ann_id"): pred for pred in base_results if pred.get("ann_id") is not None}
    refined_by_ann_id = {pred.get("ann_id"): pred for pred in refined_results if pred.get("ann_id") is not None}

    stats = defaultdict(int)
    scale_stats = defaultdict(int)
    source_by_scale = defaultdict(lambda: defaultdict(int))

    for ann in progress_iter(dataset["annotations"], total=len(dataset["annotations"]), desc="selective-sam2", disable=args.no_progress):
        ann_id = ann["id"]
        img_info = imgs_by_id[ann["image_id"]]
        base_pred = base_by_ann_id.get(ann_id)
        refined_pred = refined_by_ann_id.get(ann_id)
        if base_pred is None:
            stats["missing_base"] += 1
            continue
        validate_pred(ann, base_pred)

        use_sam2 = False
        reason = "missing_refined"
        scale_group = "unknown"
        if refined_pred is not None:
            validate_pred(ann, refined_pred)
            refine_info = refined_pred.get("sam2_refine")
            if isinstance(refine_info, dict):
                scale_group = refine_info.get("scale_group", "unknown")
            use_sam2, reason = should_use_sam2(refined_pred, args)

        chosen_pred = refined_pred if use_sam2 else base_pred
        source = "sam2" if use_sam2 else "p2seg"
        bbox_source = args.sam2_bbox_source if use_sam2 else "mask"
        ok, update_reason = update_ann_from_pred(ann, chosen_pred, img_info, source, bbox_source, args.ann_weight_mode)
        if not ok:
            stats[update_reason] += 1
            continue

        ann["selective_sam2_reason"] = reason
        ann["selective_sam2_scale_group"] = scale_group
        if args.write_refine_diagnostics and refined_pred is not None:
            compact = compact_refine_info(refined_pred.get("sam2_refine"))
            if compact is not None:
                ann["sam2_refine"] = compact
                ann["sam2_refine_selected"] = bool(compact.get("selected", False))

        stats[f"source_{source}"] += 1
        stats[f"reason_{reason}"] += 1
        scale_stats[scale_group] += 1
        source_by_scale[scale_group][source] += 1

    dump_json(dataset, args.save_ann)
    print("updated={updated}, output={output}".format(updated=stats["source_sam2"] + stats["source_p2seg"], output=args.save_ann))
    print("sources=" + json.dumps({k: stats[k] for k in sorted(stats) if k.startswith("source_")}, sort_keys=True))
    print("fallback_reasons=" + json.dumps({k.replace("reason_", ""): stats[k] for k in sorted(stats) if k.startswith("reason_")}, sort_keys=True))
    print("scale_groups=" + json.dumps(dict(sorted(scale_stats.items())), sort_keys=True))
    print("sources_by_scale=" + json.dumps({k: dict(sorted(v.items())) for k, v in sorted(source_by_scale.items())}, sort_keys=True))
    errors = {k: stats[k] for k in sorted(stats) if k in {"missing_base", "invalid_bbox", "empty_mask"}}
    if errors:
        print("errors=" + json.dumps(errors, sort_keys=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
