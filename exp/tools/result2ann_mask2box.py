import argparse
import json
import math

try:
    from pycocotools import mask as maskUtils
except ImportError:  # Keep pure validation helpers importable without COCO extras.
    maskUtils = None


def require_mask_utils():
    if maskUtils is None:
        raise ImportError(
            "pycocotools is required for mask-to-bbox conversion; install it in the active environment."
        )
    return maskUtils


def progress_iter(iterable, total=None, desc=None, disable=False):
    if disable:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def segm_to_rle(segm, height, width):
    mask_utils = require_mask_utils()
    if isinstance(segm, list):
        return mask_utils.merge(mask_utils.frPyObjects(segm, height, width))
    if isinstance(segm, dict):
        if isinstance(segm.get("counts"), list):
            return mask_utils.frPyObjects(segm, height, width)
        return segm
    raise TypeError(f"Unsupported segmentation type: {type(segm)}")


def refined_bbox_from_pred(pred):
    refine_info = pred.get("sam2_refine")
    if not isinstance(refine_info, dict):
        return None
    if not refine_info.get("selected", False):
        return None
    bbox = refine_info.get("final_bbox")
    if bbox is None:
        return None
    if len(bbox) != 4:
        raise ValueError(f"final_bbox must have four values for ann_id={pred.get('ann_id')}: {bbox}")
    return [float(v) for v in bbox]


def selected_refine_info(pred):
    refine_info = pred.get("sam2_refine")
    if isinstance(refine_info, dict) and refine_info.get("selected", False):
        return refine_info
    return None


def annotation_weight_from_pred(pred, mode):
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


def copy_refine_diagnostics(ann, pred):
    refine_info = pred.get("sam2_refine")
    if not isinstance(refine_info, dict):
        return False
    keys = [
        "gate_reason",
        "final_source",
        "final_action",
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
        "local_point_count",
        "local_same_class_point_count",
        "same_class_point_hits",
        "other_class_point_hits",
        "scale_mode",
        "scale_abs_group",
        "scale_relative_group",
        "scale_candidate_abs_group",
        "scale_ambiguity",
        "scale_head_reason",
        "scale_candidate_area",
        "scale_safe_sam_area",
        "scale_safe_feature_area",
        "router_mlp_used",
        "router_mlp_route_conf",
        "router_mlp_scale_conf",
        "router_mlp_scale_group",
        "router_mlp_reject_reason",
        "route_profile",
        "route_scale_level",
        "route_support_score",
        "route_density_penalty",
        "route_ambiguity_penalty",
        "route_effective_trust",
        "route_direct_trust_thr",
        "route_dynamic_fusion_cap",
        "route_min_prompt",
        "route_min_p2seg_iou",
        "route_min_feature_box_iou",
    ]
    ann["sam2_refine"] = {key: refine_info[key] for key in keys if key in refine_info}
    ann["sam2_refine_selected"] = bool(refine_info.get("selected", False))
    return True


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


def bbox_almost_equal(a, b, tol=1e-6):
    if a is None or b is None or len(a) != 4 or len(b) != 4:
        return False
    return all(abs(float(a[i]) - float(b[i])) <= tol for i in range(4))


def mask_area_ratio_is_valid(mask_area, source_bbox, min_ratio=0.01, max_ratio=4.0):
    """Reject only grossly degenerate masks, without consulting GT fields."""

    if source_bbox is None or len(source_bbox) != 4:
        return True
    source_area = max(0.0, float(source_bbox[2])) * max(0.0, float(source_bbox[3]))
    if source_area <= 0.0 or not math.isfinite(float(mask_area)):
        return False
    ratio = float(mask_area) / source_area
    return float(min_ratio) <= ratio <= float(max_ratio)


def summarize_base_annotations(dataset):
    annotations = dataset.get("annotations", [])
    with_true_bbox = 0
    gt_like_matches = 0
    for ann in annotations:
        bbox = ann.get("bbox")
        true_bbox = ann.get("true_bbox")
        if bbox is None or true_bbox is None:
            continue
        if len(bbox) != 4 or len(true_bbox) != 4:
            continue
        with_true_bbox += 1
        if bbox_almost_equal(bbox, true_bbox):
            gt_like_matches += 1
    gt_like_ratio = 0.0 if with_true_bbox == 0 else gt_like_matches / with_true_bbox
    return {
        "annotation_count": len(annotations),
        "with_true_bbox": with_true_bbox,
        "gt_like_matches": gt_like_matches,
        "gt_like_ratio": gt_like_ratio,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert predicted masks to pseudo bbox annotations without requiring GT masks."
    )
    parser.add_argument("ori_ann", help="Original COCO-format annotation file.")
    parser.add_argument("segm_result", help="P2Seg/P2MNet segm result json with ann_id.")
    parser.add_argument("save_ann", help="Output pseudo bbox annotation file.")
    parser.add_argument(
        "--bbox-only",
        action="store_true",
        help="Drop segmentation fields and write bbox-area only for detector retraining.",
    )
    parser.add_argument(
        "--ann-weight-mode",
        choices=["none", "score", "trust", "hybrid"],
        default="none",
        help="Write ann_weight from prediction diagnostics. Default keeps VOC detector retraining unweighted.",
    )
    parser.add_argument(
        "--write-refine-diagnostics",
        action="store_true",
        help="Copy compact SAM2 route diagnostics to each COCO annotation.",
    )
    parser.add_argument(
        "--gt-like-base-thr",
        type=float,
        default=0.98,
        help="Abort bbox-only export if most bbox fields already equal true_bbox.",
    )
    parser.add_argument(
        "--allow-gt-like-base",
        action="store_true",
        help="Allow bbox-only export even when ori_ann looks GT-like.",
    )
    parser.add_argument(
        "--mask-area-ratio-min",
        type=float,
        default=0.01,
        help="Minimum decoded-mask/source-bbox area ratio before falling back to the source bbox.",
    )
    parser.add_argument(
        "--mask-area-ratio-max",
        type=float,
        default=4.0,
        help="Maximum decoded-mask/source-bbox area ratio before falling back to the source bbox.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress display.")
    args = parser.parse_args()

    with open(args.ori_ann, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(args.segm_result, "r", encoding="utf-8") as f:
        results = json.load(f)

    anns_by_id = {ann["id"]: ann for ann in dataset["annotations"]}
    imgs_by_id = {img["id"]: img for img in dataset["images"]}
    base_summary = summarize_base_annotations(dataset)
    if (
        args.bbox_only
        and base_summary["with_true_bbox"] > 0
        and base_summary["gt_like_ratio"] >= float(args.gt_like_base_thr)
        and not args.allow_gt_like_base
    ):
        raise ValueError(
            "Detected GT-like bbox base annotation for bbox-only export: "
            f"gt_like_ratio={base_summary['gt_like_ratio']:.4f} >= {float(args.gt_like_base_thr):.4f}. "
            "Use a pseudo-box base annotation file or pass --allow-gt-like-base to override."
        )

    updated = 0
    missing_ann_id = 0
    empty_masks = 0
    invalid_bboxes = 0
    clipped_bboxes = 0
    guarded_masks = 0
    dropped_segmentations = 0
    wrote_refine_diagnostics = 0
    source_counts = {}
    scale_counts = {}
    seen_pred_ann_ids = set()
    selected_refine_bbox_ann_ids = set()

    for pred in progress_iter(results, total=len(results), desc="mask2box", disable=args.no_progress):
        ann_id = pred.get("ann_id")
        if ann_id is None or ann_id not in anns_by_id:
            missing_ann_id += 1
            continue
        seen_pred_ann_ids.add(ann_id)

        ori_ann = anns_by_id[ann_id]
        if ori_ann["image_id"] != pred["image_id"]:
            raise AssertionError(f"image_id mismatch for ann_id={ann_id}")
        if ori_ann["category_id"] != pred["category_id"]:
            raise AssertionError(f"category_id mismatch for ann_id={ann_id}")

        img_info = imgs_by_id[pred["image_id"]]
        rle = segm_to_rle(pred["segmentation"], img_info["height"], img_info["width"])
        mask_utils = require_mask_utils()
        area = float(mask_utils.area(rle))
        bbox = refined_bbox_from_pred(pred)
        bbox_from_decoder = bbox is not None
        if bbox is None:
            source_bbox = sanitize_xywh_bbox(pred.get("bbox"), img_info["width"], img_info["height"])
            if not mask_area_ratio_is_valid(area, source_bbox, args.mask_area_ratio_min, args.mask_area_ratio_max) and source_bbox is not None:
                bbox = source_bbox
                area = float(source_bbox[2] * source_bbox[3])
                guarded_masks += 1
            else:
                bbox = mask_utils.toBbox(rle).tolist()
        elif bbox[2] > 0 and bbox[3] > 0:
            area = float(bbox[2] * bbox[3])

        if area <= 0 or bbox[2] <= 0 or bbox[3] <= 0:
            empty_masks += 1
            if "bbox" not in pred:
                continue
            bbox = pred["bbox"]
            area = float(bbox[2] * bbox[3])
            bbox_from_decoder = False

        clean_bbox = sanitize_xywh_bbox(bbox, img_info["width"], img_info["height"])
        if clean_bbox is None:
            invalid_bboxes += 1
            continue
        if any(abs(float(bbox[i]) - clean_bbox[i]) > 1e-6 for i in range(4)):
            clipped_bboxes += 1

        ori_ann["bbox"] = clean_bbox
        ori_ann["area"] = float(clean_bbox[2] * clean_bbox[3])
        if args.bbox_only:
            if "segmentation" in ori_ann:
                dropped_segmentations += 1
                ori_ann.pop("segmentation", None)
        else:
            ori_ann["segmentation"] = pred["segmentation"]
        ann_weight = annotation_weight_from_pred(pred, args.ann_weight_mode)
        if ann_weight is not None:
            ori_ann["ann_weight"] = float(ann_weight)
        elif "ann_weight" in ori_ann:
            ori_ann.pop("ann_weight", None)
        refine_info = selected_refine_info(pred)
        if bbox_from_decoder and refine_info is not None:
            selected_refine_bbox_ann_ids.add(ann_id)
            source = refine_info.get("final_source", "decoder")
            scale = refine_info.get("scale_group", "unknown")
            ori_ann["pseudo_bbox_source"] = source
            ori_ann["pseudo_bbox_scale_group"] = scale
            source_counts[source] = source_counts.get(source, 0) + 1
            scale_counts[scale] = scale_counts.get(scale, 0) + 1
        if args.write_refine_diagnostics:
            if copy_refine_diagnostics(ori_ann, pred):
                wrote_refine_diagnostics += 1
        updated += 1

    total_annotations = len(dataset["annotations"])
    missing_prediction_ann_ids = set(anns_by_id) - seen_pred_ann_ids
    missing_prediction_gt_like = 0
    missing_prediction_with_true_bbox = 0
    for ann_id in missing_prediction_ann_ids:
        ann = anns_by_id[ann_id]
        true_bbox = ann.get("true_bbox")
        bbox = ann.get("bbox")
        if true_bbox is None or bbox is None:
            continue
        if len(true_bbox) != 4 or len(bbox) != 4:
            continue
        missing_prediction_with_true_bbox += 1
        if bbox_almost_equal(bbox, true_bbox):
            missing_prediction_gt_like += 1

    if args.bbox_only:
        for ann in dataset["annotations"]:
            if args.ann_weight_mode == "none":
                ann.pop("ann_weight", None)
            if "segmentation" in ann:
                dropped_segmentations += 1
                ann.pop("segmentation", None)
            bbox = sanitize_xywh_bbox(ann.get("bbox"), imgs_by_id[ann["image_id"]]["width"], imgs_by_id[ann["image_id"]]["height"])
            if bbox is None:
                invalid_bboxes += 1
                ann["ignore"] = 1
                ann["area"] = 0.0
                continue
            ann["bbox"] = bbox
            ann["area"] = float(bbox[2] * bbox[3])
            if ann["id"] in missing_prediction_ann_ids:
                ann["pseudo_bbox_source"] = "fallback_ori_ann"
                ann["pseudo_bbox_scale_group"] = ann.get("pseudo_bbox_scale_group", "unknown")
                source_counts["fallback_ori_ann"] = source_counts.get("fallback_ori_ann", 0) + 1

    with open(args.save_ann, "w", encoding="utf-8") as f:
        json.dump(dataset, f)

    print(
        "updated={updated}, missing_ann_id={missing_ann_id}, empty_masks={empty_masks}, "
        "invalid_bboxes={invalid_bboxes}, clipped_bboxes={clipped_bboxes}, "
        "guarded_masks={guarded_masks}, dropped_segmentations={dropped_segmentations}, "
        "wrote_refine_diagnostics={wrote_refine_diagnostics}".format(
            updated=updated,
            missing_ann_id=missing_ann_id,
            empty_masks=empty_masks,
            invalid_bboxes=invalid_bboxes,
            clipped_bboxes=clipped_bboxes,
            guarded_masks=guarded_masks,
            dropped_segmentations=dropped_segmentations,
            wrote_refine_diagnostics=wrote_refine_diagnostics,
        )
    )
    coverage_summary = {
        "annotation_count": total_annotations,
        "matched_prediction_annotations": len(seen_pred_ann_ids),
        "missing_prediction_annotations": len(missing_prediction_ann_ids),
        "missing_prediction_ratio": 0.0 if total_annotations == 0 else len(missing_prediction_ann_ids) / total_annotations,
        "selected_refine_bbox_annotations": len(selected_refine_bbox_ann_ids),
        "selected_refine_bbox_ratio": 0.0 if total_annotations == 0 else len(selected_refine_bbox_ann_ids) / total_annotations,
        "prediction_annotations_without_selected_refine_bbox": len(seen_pred_ann_ids - selected_refine_bbox_ann_ids),
        "prediction_annotations_without_selected_refine_bbox_ratio": 0.0
        if total_annotations == 0
        else len(seen_pred_ann_ids - selected_refine_bbox_ann_ids) / total_annotations,
    }
    fallback_summary = {
        "missing_prediction_annotations": len(missing_prediction_ann_ids),
        "missing_prediction_with_true_bbox": missing_prediction_with_true_bbox,
        "missing_prediction_gt_like": missing_prediction_gt_like,
        "missing_prediction_gt_like_ratio": 0.0
        if missing_prediction_with_true_bbox == 0
        else missing_prediction_gt_like / missing_prediction_with_true_bbox,
    }
    print("base_annotation_summary=" + json.dumps(base_summary, sort_keys=True))
    print("prediction_coverage_summary=" + json.dumps(coverage_summary, sort_keys=True))
    print("fallback_summary=" + json.dumps(fallback_summary, sort_keys=True))
    if source_counts:
        print("pseudo_bbox_sources=" + json.dumps(dict(sorted(source_counts.items())), sort_keys=True))
    if scale_counts:
        print("pseudo_bbox_scale_groups=" + json.dumps(dict(sorted(scale_counts.items())), sort_keys=True))


if __name__ == "__main__":
    main()
