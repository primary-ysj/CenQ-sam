import argparse
import copy
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


maskUtils = None

ROUTER_ROUTE_LABELS = {"p2seg": 0, "fusion": 1, "sam2": 2, "feature": 3}
ROUTER_ROUTE_NAMES = {v: k for k, v in ROUTER_ROUTE_LABELS.items()}
ROUTER_SCALE_LABELS = {"small": 0, "medium": 1, "large": 2}
ROUTER_SCALE_NAMES = {v: k for k, v in ROUTER_SCALE_LABELS.items()}
ROUTE_PROFILE_OVERRIDES = {
    "baseline": {},
    "v0.3": {
        "medium_gate_min_score_margin": 0.015,
        "large_gate_min_score_margin": 0.008,
        "medium_direct_sam_trust_thr": 0.76,
        "medium_direct_sam_min_prompt": 0.55,
        "medium_direct_sam_min_p2seg_iou": 0.30,
        "medium_direct_sam_min_feature_box_iou": 0.12,
        "large_direct_sam_trust_thr": 0.78,
        "large_direct_sam_min_prompt": 0.50,
        "large_direct_sam_min_p2seg_iou": 0.24,
    },
}
ROUTER_FEATURE_KEYS = [
    "sam_score",
    "total_score",
    "score_margin",
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
]
ROUTER_FEATURE_KEYS_V2 = ROUTER_FEATURE_KEYS + [
    "p2seg_log_area_norm",
    "candidate_log_area_norm",
    "safe_sam_log_area_norm",
    "safe_feature_log_area_norm",
    "p2seg_aspect_ratio",
]
ROUTER_FEATURE_KEYS_BY_VERSION = {
    1: ROUTER_FEATURE_KEYS,
    2: ROUTER_FEATURE_KEYS_V2,
}


class SemanticProposalRanker:
    def __init__(self, checkpoint_path, device):
        import torch
        import torch.nn as nn
        from torchvision.models import ResNet50_Weights, resnet50

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.categories = {category_id: index for index, category_id in enumerate(checkpoint["categories"])}
        self.device = device
        self.transforms = ResNet50_Weights.IMAGENET1K_V2.transforms()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(2048, len(self.categories)).to(device).eval()
        self.head.load_state_dict(checkpoint["head"])
        for module in (self.backbone, self.head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def score(self, image, mask, category_id):
        import torch

        label = self.categories.get(category_id)
        box = mask_to_xyxy(mask)
        if label is None or box is None:
            return 0.0
        crop = Image.fromarray(image).crop([float(v) for v in box])
        if crop.width < 2 or crop.height < 2:
            return 0.0
        with torch.no_grad():
            feature = self.backbone(self.transforms(crop).unsqueeze(0).to(self.device))
            return float(torch.softmax(self.head(feature), dim=1)[0, label])


def progress_iter(iterable, total=None, desc=None, disable=False):
    if disable:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def apply_route_profile(args):
    profile = getattr(args, "route_profile", "baseline") or "baseline"
    if profile in {"baseline", "default"}:
        return args
    overrides = ROUTE_PROFILE_OVERRIDES.get(profile)
    if overrides is None:
        raise ValueError(f"Unsupported route profile: {profile}")
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def active_ablation(args):
    return getattr(args, "ablation", "full")


SAGE_ABLATIONS = {
    "g1_low_sam_score",
    "g2_low_cross_source_iou",
    "g3_low_prompt_stability",
    "sage_box",
}


def is_sage_ablation(args):
    return active_ablation(args) in SAGE_ABLATIONS


def is_reliability_free(args):
    return active_ablation(args) in {
        "no_reliability_cues",
        "scpr",
        "uniform_fusion",
        "all_direct_sam",
        "no_scale_conditioning",
        "no_small_anchor",
        "no_medium_large_direct",
    } or is_sage_ablation(args)


def uses_semantic_candidate_ranking(args):
    return active_ablation(args) == "all_direct_sam" and bool(getattr(args, "semantic_candidate_ranking", False))


def bypass_small_anchor(args):
    return active_ablation(args) in {
        "no_small_safety",
        "no_scale_cues",
        "all_direct_sam",
        "no_scale_conditioning",
        "no_small_anchor",
    } or is_sage_ablation(args)


def sage_policy(args, score, sam_box, p2seg_box):
    """Select a source using only frozen, GT-free candidate diagnostics."""
    if sam_box is None:
        return "p2seg", ["missing_sam_box"]
    if p2seg_box is None:
        return "sam2", ["missing_p2seg_box"]

    mode = active_ablation(args)
    reasons = []
    if mode in {"g1_low_sam_score", "sage_box"} and float(score.get("sam_score", 0.0)) < float(args.sage_low_sam_score):
        reasons.append("low_sam_score")
    if mode in {"g2_low_cross_source_iou", "sage_box"} and float(score.get("p2seg_iou", 0.0)) < float(args.sage_low_cross_source_iou):
        reasons.append("low_cross_source_iou")
    if mode in {"g3_low_prompt_stability", "sage_box"} and float(score.get("prompt_consistency_score", 0.0)) < float(args.sage_low_prompt_stability):
        reasons.append("low_prompt_stability")
    return ("p2seg", reasons) if reasons else ("sam2", ["sam2_selected"])


def gate_margin_threshold(score, args):
    default = float(getattr(args, "gate_min_score_margin", 0.05))
    scale_group = score.get("routing_scale_group", score.get("scale_group", "unknown"))
    scale_key = {
        "small": "small_gate_min_score_margin",
        "medium": "medium_gate_min_score_margin",
        "large": "large_gate_min_score_margin",
    }.get(scale_group)
    if scale_key is None:
        return default
    threshold = float(getattr(args, scale_key, default))
    ambiguity_penalty = scale_ambiguity_penalty(score.get("scale_ambiguity", "none"), scale_group)
    density_penalty = clamp(finite_float(score.get("local_point_density", 0.0)))
    route_scale_level = finite_float(score.get("route_scale_level", score.get("scale_prob", 0.0)), 0.0)
    if scale_group == "medium":
        progress = clamp(route_scale_level)
        large_margin = float(getattr(args, "large_gate_min_score_margin", threshold))
        threshold = threshold - (threshold - large_margin) * 0.75 * progress
    elif scale_group == "large":
        progress = clamp(route_scale_level - 1.0)
        threshold = threshold * (1.0 - 0.20 * progress)
    threshold += 0.010 * ambiguity_penalty + 0.005 * density_penalty
    return max(0.005, float(threshold))


def get_mask_utils():
    global maskUtils
    if maskUtils is None:
        from pycocotools import mask as _mask_utils

        maskUtils = _mask_utils
    return maskUtils


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_semantic_boxes(path):
    if not path:
        return {}
    raw = load_json(path)
    rows = raw.get("boxes", raw) if isinstance(raw, dict) else {}
    loaded = {}
    for ann_id, row in rows.items():
        if not isinstance(row, dict):
            continue
        box = row.get("bbox_xyxy", row.get("box"))
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            loaded[int(ann_id)] = {
                "box": np.asarray(box, dtype=np.float32),
                "score": float(row.get("score", 0.0)),
            }
        except (TypeError, ValueError):
            continue
    return loaded


def dump_json(obj, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(obj, f)


def softmax(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    arr = arr - float(arr.max())
    exp = np.exp(arr)
    return exp / max(float(exp.sum()), 1e-12)


def finite_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def segm_to_rle(segm, height, width):
    mask_utils = get_mask_utils()
    if isinstance(segm, list):
        return mask_utils.merge(mask_utils.frPyObjects(segm, height, width))
    if isinstance(segm, dict):
        if isinstance(segm.get("counts"), list):
            return mask_utils.frPyObjects(segm, height, width)
        return segm
    raise TypeError(f"Unsupported segmentation type: {type(segm)}")


def decode_segm(segm, height, width):
    mask_utils = get_mask_utils()
    rle = segm_to_rle(segm, height, width)
    return mask_utils.decode(rle).astype(bool)


def encode_mask(mask):
    mask_utils = get_mask_utils()
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def mask_area_from_pred(pred, img_info):
    try:
        mask_utils = get_mask_utils()
        rle = segm_to_rle(pred["segmentation"], img_info["height"], img_info["width"])
        area = float(mask_utils.area(rle))
    except Exception:
        area = 0.0
    if area <= 0 and "bbox" in pred:
        area = float(pred["bbox"][2] * pred["bbox"][3])
    return max(area, 0.0)


def bbox_xywh_to_xyxy(bbox):
    x, y, w, h = [float(v) for v in bbox]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def bbox_xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def clip_box(box, width, height):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), float(width - 1))
    y1 = min(max(y1, 0.0), float(height - 1))
    x2 = min(max(x2, x1 + 1.0), float(width))
    y2 = min(max(y2, y1 + 1.0), float(height))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def expand_box(box, ratio, width, height):
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    nw = max(1.0, w * (1.0 + ratio))
    nh = max(1.0, h * (1.0 + ratio))
    return clip_box([cx - nw * 0.5, cy - nh * 0.5, cx + nw * 0.5, cy + nh * 0.5], width, height)


def mask_to_xyxy(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def bbox_iou_xyxy(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return 0.0 if union <= 0 else inter / union


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else float(inter) / float(union)


def mine_prompt_consensus(scored_candidates, point, args):
    """Aggregate compatible SAM2 prompts into a point-connected candidate."""
    if active_ablation(args) == "no_prompt_consensus" or not bool(getattr(args, "enable_prompt_consensus", False)) or len(scored_candidates) < 2:
        return None

    ranked = sorted(scored_candidates, key=lambda item: item[1]["total"], reverse=True)
    anchor, anchor_score = ranked[0]
    anchor_mask = anchor["mask"]
    anchor_area = max(float(anchor_mask.sum()), 1.0)
    members = [(anchor, anchor_score)]
    for candidate, score in ranked[1:int(getattr(args, "consensus_topk", 3))]:
        candidate_mask = candidate["mask"]
        overlap = mask_iou(anchor_mask, candidate_mask)
        containment = float(np.logical_and(anchor_mask, candidate_mask).sum()) / max(float(min(anchor_mask.sum(), candidate_mask.sum())), 1.0)
        if overlap >= float(getattr(args, "consensus_min_iou", 0.45)) or containment >= float(getattr(args, "consensus_min_containment", 0.80)):
            members.append((candidate, score))

    if len(members) < 2:
        return None
    totals = np.asarray([score["total"] for _, score in members], dtype=np.float32)
    temperature = max(float(getattr(args, "consensus_temperature", 0.20)), 1e-6)
    weights = softmax((totals - float(totals.max())) / temperature)
    support = np.zeros_like(anchor_mask, dtype=np.float32)
    for weight, (candidate, _) in zip(weights, members):
        support += float(weight) * candidate["mask"].astype(np.float32)
    consensus_mask = support >= float(getattr(args, "consensus_min_support", 0.55))
    if point is None or not point_inside_mask(consensus_mask, point):
        return None
    consensus_mask = component_from_point(consensus_mask, point)
    if consensus_mask.sum() <= 1:
        return None
    expansion = float(consensus_mask.sum()) / anchor_area
    if expansion > float(getattr(args, "consensus_max_expansion", 1.50)):
        return None

    weighted_iou = float(np.average([mask_iou(anchor_mask, candidate["mask"]) for candidate, _ in members[1:]], weights=weights[1:]))
    return {
        "mask": consensus_mask,
        "sam_score": float(np.dot(weights, [candidate.get("sam_score", 0.0) for candidate, _ in members])),
        "prompt": "prompt_consensus",
        "consensus_member_count": len(members),
        "consensus_weighted_iou": weighted_iou,
        "consensus_expansion": expansion,
    }


def calibrate_box_with_feature(sam_box, feature_box, score, scale_group, img_info, args):
    """Use a point-connected feature component to make a conservative boundary correction."""
    info = {
        "boundary_calibration_applied": False,
        "boundary_calibration_weight": 0.0,
        "raw_sam_box": None if sam_box is None else [float(v) for v in sam_box],
        "boundary_calibrated_box": None if sam_box is None else [float(v) for v in sam_box],
    }
    if active_ablation(args) == "no_boundary_calibration" or not bool(getattr(args, "enable_boundary_calibration", False)) or scale_group == "small":
        return sam_box, info
    if sam_box is None or feature_box is None or not score.get("feature_available", False):
        return sam_box, info
    feature_score = float(score.get("feature_consistency_score", 0.0))
    min_score = float(getattr(args, "boundary_calibration_min_feature_score", 0.70))
    if feature_score < min_score or float(score.get("feature_box_iou", 0.0)) < float(getattr(args, "boundary_calibration_min_box_iou", 0.30)):
        return sam_box, info
    sam_area = max(box_area_xyxy(sam_box), 1.0)
    feature_area = max(box_area_xyxy(feature_box), 1.0)
    area_ratio = max(sam_area, feature_area) / min(sam_area, feature_area)
    if area_ratio > float(getattr(args, "boundary_calibration_max_area_ratio", 2.0)):
        return sam_box, info
    confidence = clamp((feature_score - min_score) / max(1.0 - min_score, 1e-6))
    max_weight = float(getattr(args, "boundary_calibration_max_weight", 0.35))
    min_weight = float(getattr(args, "boundary_calibration_min_weight", 0.08))
    weight = min(max_weight, max(min_weight, max_weight * confidence))
    calibrated = clip_box(lerp_box_xyxy(sam_box, feature_box, weight), img_info["width"], img_info["height"])
    info.update({
        "boundary_calibration_applied": True,
        "boundary_calibration_weight": float(weight),
        "boundary_calibrated_box": [float(v) for v in calibrated],
    })
    return calibrated, info


def color_refine_candidate(candidate, image, point, args):
    """Create a point-anchored GrabCut candidate from a SAM2 mask without changing its prompt."""
    if not bool(getattr(args, "enable_color_boundary_refine", False)) or image is None or point is None:
        return None
    try:
        import cv2
    except ImportError:
        return None
    raw_mask = candidate["mask"].astype(bool)
    raw_box = mask_to_xyxy(raw_mask)
    if raw_box is None:
        return None
    h, w = raw_mask.shape
    roi = expand_box(raw_box, float(getattr(args, "color_refine_roi_expand", 0.15)), w, h)
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[y1:y2, x1:x2] = cv2.GC_PR_BGD
    gc_mask[raw_mask] = cv2.GC_PR_FGD
    px, py = int(round(point[0])), int(round(point[1]))
    if px < 0 or py < 0 or px >= w or py >= h:
        return None
    radius = max(1, int(getattr(args, "color_refine_point_radius", 2)))
    cv2.circle(gc_mask, (px, py), radius, cv2.GC_FGD, thickness=-1)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image.copy(), gc_mask, None, bgd_model, fgd_model, int(getattr(args, "color_refine_iterations", 2)), cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    refined_mask = np.logical_or(gc_mask == cv2.GC_FGD, gc_mask == cv2.GC_PR_FGD)
    if not point_inside_mask(refined_mask, point):
        return None
    refined_mask = component_from_point(refined_mask, point)
    raw_area = max(float(raw_mask.sum()), 1.0)
    area_ratio = float(refined_mask.sum()) / raw_area
    overlap = mask_iou(raw_mask, refined_mask)
    if overlap < float(getattr(args, "color_refine_min_mask_iou", 0.60)):
        return None
    if not float(getattr(args, "color_refine_min_area_ratio", 0.50)) <= area_ratio <= float(getattr(args, "color_refine_max_area_ratio", 1.50)):
        return None
    return {
        "mask": refined_mask,
        "sam_score": clamp(float(candidate.get("sam_score", 0.0)) + float(getattr(args, "color_refine_score_bonus", 0.01))),
        "prompt": "color_boundary_refine",
        "color_refine_mask_iou": overlap,
        "color_refine_area_ratio": area_ratio,
    }


def point_from_ann(ann):
    if "point" in ann and ann["point"] is not None:
        return [float(ann["point"][0]), float(ann["point"][1])]
    if "points" in ann and ann["points"]:
        pt = ann["points"][0]
        return [float(pt[0]), float(pt[1])]
    if "bbox" in ann:
        x, y, w, h = [float(v) for v in ann["bbox"]]
        return [x + w * 0.5, y + h * 0.5]
    return None


def point_inside_mask(mask, point):
    if point is None:
        return False
    x, y = int(round(point[0])), int(round(point[1]))
    if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
        return False
    return bool(mask[y, x])


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def scale_ambiguity_penalty(ambiguity, scale_group="unknown"):
    if ambiguity in (None, "", "none"):
        return 0.0
    base = {
        "boundary": 0.20,
        "rel_large_abs_medium": 0.10,
        "rel_small_abs_medium": 0.80,
        "abs_small_but_credible_medium": 0.65,
        "abs_medium_but_credible_large": 0.20,
        "abs_small_but_credible_large": 0.55,
        "rel_medium_abs_large": 0.30,
        "rel_small_abs_large": 0.40,
        "rel_medium_abs_small": 0.65,
        "rel_large_abs_small": 0.75,
    }.get(str(ambiguity), 0.30)
    if scale_group == "small":
        base = min(1.0, base + 0.10)
    elif scale_group == "large" and ambiguity in {"boundary", "rel_large_abs_medium", "abs_medium_but_credible_large"}:
        base *= 0.75
    return clamp(base)


def box_area_xyxy(box):
    if box is None:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center_xyxy(box):
    if box is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def union_box_xyxy(a, b):
    if a is None:
        return None if b is None else np.asarray(b, dtype=np.float32)
    if b is None:
        return np.asarray(a, dtype=np.float32)
    return np.array([
        min(float(a[0]), float(b[0])),
        min(float(a[1]), float(b[1])),
        max(float(a[2]), float(b[2])),
        max(float(a[3]), float(b[3])),
    ], dtype=np.float32)


def lerp_box_xyxy(a, b, weight_b):
    if a is None:
        return None if b is None else np.asarray(b, dtype=np.float32)
    if b is None:
        return np.asarray(a, dtype=np.float32)
    weight_b = clamp(weight_b)
    return np.asarray(a, dtype=np.float32) * (1.0 - weight_b) + np.asarray(b, dtype=np.float32) * weight_b


def resize_bool_mask(mask, out_shape):
    out_h, out_w = int(out_shape[0]), int(out_shape[1])
    if mask.shape[0] == out_h and mask.shape[1] == out_w:
        return mask.astype(bool)
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    img = img.resize((out_w, out_h), Image.NEAREST)
    return np.asarray(img) > 0


def scale_point(point, src_width, src_height, dst_width, dst_height):
    if point is None or src_width <= 0 or src_height <= 0:
        return None
    x = int(round(float(point[0]) * max(dst_width - 1, 0) / max(src_width - 1, 1)))
    y = int(round(float(point[1]) * max(dst_height - 1, 0) / max(src_height - 1, 1)))
    return [min(max(x, 0), dst_width - 1), min(max(y, 0), dst_height - 1)]


def component_from_point(mask, point_xy):
    if point_xy is None or mask.size == 0:
        return np.zeros_like(mask, dtype=bool)
    px, py = int(point_xy[0]), int(point_xy[1])
    if py < 0 or py >= mask.shape[0] or px < 0 or px >= mask.shape[1]:
        return np.zeros_like(mask, dtype=bool)
    if not mask[py, px]:
        mask = mask.copy()
        mask[py, px] = True

    out = np.zeros_like(mask, dtype=bool)
    stack = [(py, px)]
    out[py, px] = True
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or ny >= mask.shape[0] or nx < 0 or nx >= mask.shape[1]:
                continue
            if mask[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                stack.append((ny, nx))
    return out


def feature_box_to_image_box(box, feat_width, feat_height, image_width, image_height):
    if box is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    sx = float(image_width) / max(float(feat_width), 1.0)
    sy = float(image_height) / max(float(feat_height), 1.0)
    return clip_box([x1 * sx, y1 * sy, x2 * sx, y2 * sy], image_width, image_height)


def image_box_to_feature_box(box, image_width, image_height, feat_width, feat_height):
    if box is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    sx = float(feat_width) / max(float(image_width), 1.0)
    sy = float(feat_height) / max(float(image_height), 1.0)
    return np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)


def expand_box_mask(box, shape, ratio):
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w), dtype=bool)
    if box is None:
        return out
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    x1 = int(math.floor(cx - bw * (1.0 + ratio) * 0.5))
    y1 = int(math.floor(cy - bh * (1.0 + ratio) * 0.5))
    x2 = int(math.ceil(cx + bw * (1.0 + ratio) * 0.5))
    y2 = int(math.ceil(cy + bh * (1.0 + ratio) * 0.5))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = True
    return out


def make_mask_prompt(mask):
    prompt = Image.fromarray((mask.astype(np.uint8) * 255))
    prompt = prompt.resize((256, 256), Image.BILINEAR)
    prompt = np.asarray(prompt, dtype=np.float32) / 255.0
    prompt = np.where(prompt > 0.5, 10.0, -10.0).astype(np.float32)
    return prompt[None, :, :]


def parse_float_list(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_prompt_modes(value):
    return {v.strip() for v in value.split(",") if v.strip()}


def nearest_negative_points(target_point, other_anns, anns_by_id, mode, max_points):
    if max_points <= 0 or mode == "none" or target_point is None:
        return []
    candidates = []
    target_ann = anns_by_id.get(target_point[2]) if len(target_point) > 2 else None
    target_category = target_ann.get("category_id") if target_ann else None
    tx, ty = target_point[0], target_point[1]
    for ann in other_anns:
        if mode == "same-class" and ann.get("category_id") != target_category:
            continue
        pt = point_from_ann(ann)
        if pt is None:
            continue
        dist = (pt[0] - tx) ** 2 + (pt[1] - ty) ** 2
        candidates.append((dist, pt, ann.get("category_id")))
    candidates.sort(key=lambda x: x[0])
    return [(pt, category_id) for _, pt, category_id in candidates[:max_points]]


def stats(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return None
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(max(arr.std(), 1e-6)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def percentile_stats(values, percentiles):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return None
    out = stats(arr)
    for p in percentiles:
        out[f"p{int(round(p)):02d}"] = float(np.percentile(arr, p))
    return out


def build_online_priors(results, images_by_id, min_pair_count):
    class_values = defaultdict(list)
    rel_values = defaultdict(list)
    area_by_ann_id = {}
    results_by_image = defaultdict(list)

    for pred in results:
        img = images_by_id.get(pred.get("image_id"))
        ann_id = pred.get("ann_id")
        if img is None or ann_id is None:
            continue
        area = mask_area_from_pred(pred, img)
        area_by_ann_id[ann_id] = area
        image_area = float(img["height"] * img["width"])
        if area > 0 and image_area > 0:
            class_values[str(pred["category_id"])].append(math.log(area / image_area + 1e-12))
            results_by_image[pred["image_id"]].append(pred)

    for preds in results_by_image.values():
        for pred_a in preds:
            ann_a = pred_a.get("ann_id")
            area_a = area_by_ann_id.get(ann_a, 0.0)
            if area_a <= 0:
                continue
            for pred_b in preds:
                if pred_a is pred_b:
                    continue
                ann_b = pred_b.get("ann_id")
                area_b = area_by_ann_id.get(ann_b, 0.0)
                if area_b <= 0:
                    continue
                key = f"{pred_a['category_id']}|{pred_b['category_id']}"
                rel_values[key].append(math.log(area_a / area_b + 1e-12))

    priors = {
        "class_area": {k: stats(v) for k, v in class_values.items()},
        "relative_area": {k: stats(v) for k, v in rel_values.items() if len(v) >= min_pair_count},
        "area_by_ann_id": area_by_ann_id,
    }
    return priors


def build_scale_priors(results, images_by_id, small_quantile, large_quantile):
    class_values = defaultdict(list)
    global_values = []
    area_by_ann_id = {}

    for pred in results:
        img = images_by_id.get(pred.get("image_id"))
        ann_id = pred.get("ann_id")
        if img is None:
            continue
        area = float(pred.get("_p2seg_area", 0.0))
        if area <= 0:
            area = mask_area_from_pred(pred, img)
        if ann_id is not None:
            area_by_ann_id[ann_id] = area
        image_area = float(img["height"] * img["width"])
        if area <= 0 or image_area <= 0:
            continue
        value = math.log(area / image_area + 1e-12)
        global_values.append(value)
        class_values[str(pred["category_id"])].append(value)

    percentiles = [small_quantile, large_quantile]
    return {
        "mode": "log_area_ratio",
        "small_quantile": float(small_quantile),
        "large_quantile": float(large_quantile),
        "global": percentile_stats(global_values, percentiles),
        "class": {k: percentile_stats(v, percentiles) for k, v in class_values.items()},
        "area_by_ann_id": area_by_ann_id,
    }


def load_priors(path):
    if not path:
        return {"class_area": {}, "relative_area": {}, "area_by_ann_id": {}, "scale": {}}
    priors = load_json(path)
    priors.setdefault("class_area", {})
    priors.setdefault("relative_area", {})
    priors.setdefault("area_by_ann_id", {})
    priors.setdefault("scale", {})
    return priors


def stat_score(value, stat):
    if not stat:
        return 0.0
    z = abs((value - float(stat["mean"])) / max(float(stat.get("std", 1e-6)), 1e-6))
    return float(math.exp(-0.5 * z * z))


def class_area_score(mask_area, image_area, category_id, priors):
    if mask_area <= 0 or image_area <= 0:
        return 0.0
    stat = priors.get("class_area", {}).get(str(category_id))
    return stat_score(math.log(mask_area / image_area + 1e-12), stat)


def relative_size_score(mask_area, category_id, image_preds, current_ann_id, priors):
    scores = []
    area_by_ann_id = priors.get("area_by_ann_id", {})
    if mask_area <= 0:
        return 0.0
    for other in image_preds:
        other_ann_id = other.get("ann_id")
        if other_ann_id == current_ann_id:
            continue
        other_area = area_by_ann_id.get(other_ann_id)
        if other_area is None:
            other_area = other.get("_p2seg_area", 0.0)
        if other_area <= 0:
            continue
        key = f"{category_id}|{other['category_id']}"
        stat = priors.get("relative_area", {}).get(key)
        if not stat:
            continue
        scores.append(stat_score(math.log(mask_area / other_area + 1e-12), stat))
    return 0.0 if not scores else float(np.mean(scores))


class ScaleAwareRouter:
    def __init__(self, args, priors):
        self.args = args
        self.scale_priors = priors.get("scale", {}) if priors else {}
        self.mlp = self._load_mlp(args.router_mlp_json)

    def _load_mlp(self, path):
        if not path:
            return None
        model = load_json(path)
        required = {"feature_keys", "mean", "std", "hidden_weight", "hidden_bias"}
        missing = sorted(required.difference(model.keys()))
        if missing:
            raise ValueError(f"router MLP missing keys: {missing}")
        version = int(model.get("router_head_version", 1))
        expected_keys = ROUTER_FEATURE_KEYS_BY_VERSION.get(version)
        if expected_keys is None:
            raise ValueError(f"Unsupported router MLP version: {version}")
        if model["feature_keys"] != expected_keys:
            raise ValueError(f"router MLP feature order mismatch: {model['feature_keys']} != {expected_keys}")
        if version == 1:
            v1_required = {"scale_weight", "scale_bias", "route_weight", "route_bias", "fusion_weight", "fusion_bias"}
            missing = sorted(v1_required.difference(model.keys()))
            if missing:
                raise ValueError(f"router V1 MLP missing keys: {missing}")
        else:
            v2_required = {"scale_weight", "scale_bias", "release_weight", "release_bias", "fusion_delta_weight", "fusion_delta_bias"}
            missing = sorted(v2_required.difference(model.keys()))
            if missing:
                raise ValueError(f"router V2 MLP missing keys: {missing}")
        model["router_head_version"] = version
        return model

    def group_from_area(self, area, category_id, image_area, score=None, sam_box=None, feature_box=None):
        relative_group, relative_prob, relative_meta = self._relative_group_from_area(area, category_id, image_area)
        abs_group = self._absolute_group_from_area(area)
        mode = getattr(self.args, "scale_head_mode", "calibrated")
        if mode == "relative":
            relative_meta.update({
                "scale_abs_group": abs_group,
                "scale_relative_group": relative_group,
                "scale_ambiguity": "none",
                "scale_head_reason": "relative_mode",
            })
            return relative_group, relative_prob, relative_meta
        if mode == "absolute":
            return abs_group, self._abs_scale_prob(area), {
                "scale_mode": "absolute",
                "scale_abs_group": abs_group,
                "scale_relative_group": relative_group,
                "scale_ambiguity": "none",
                "scale_head_reason": "absolute_mode",
                "scale_log_area": self._log_area(area, image_area),
                "scale_small_thr": float(getattr(self.args, "scale_small_abs_thr", self.args.small_area_thr)),
                "scale_large_thr": float(getattr(self.args, "scale_large_abs_thr", self.args.large_area_thr)),
                "scale_candidate_area": float(max(area, 0.0)),
                "scale_safe_sam_area": 0.0,
                "scale_safe_feature_area": 0.0,
            }
        return self._calibrated_group_from_area(
            area, image_area, score or {}, sam_box, feature_box, relative_group, relative_meta)

    def _relative_group_from_area(self, area, category_id, image_area):
        if area <= 0 or image_area <= 0:
            return scale_group_from_area(area, self.args), 0.0, {
                "scale_mode": "absolute",
                "scale_small_thr": float(self.args.small_area_thr),
                "scale_large_thr": float(self.args.large_area_thr),
            }
        if not self.args.use_adaptive_scale_priors or not self.scale_priors:
            group = scale_group_from_area(area, self.args)
            return group, self._absolute_scale_prob(area), {
                "scale_mode": "absolute",
                "scale_small_thr": float(self.args.small_area_thr),
                "scale_large_thr": float(self.args.large_area_thr),
            }

        value = math.log(area / image_area + 1e-12)
        stat, source = self._select_scale_stat(category_id)
        if not stat:
            group = scale_group_from_area(area, self.args)
            return group, self._absolute_scale_prob(area), {
                "scale_mode": "absolute_fallback",
                "scale_small_thr": float(self.args.small_area_thr),
                "scale_large_thr": float(self.args.large_area_thr),
            }

        small_key = f"p{int(round(float(self.args.scale_small_quantile))):02d}"
        large_key = f"p{int(round(float(self.args.scale_large_quantile))):02d}"
        small_thr = float(stat.get(small_key, stat.get("p33", stat.get("p50", -12.0))))
        large_thr = float(stat.get(large_key, stat.get("p67", stat.get("p50", -8.0))))
        if large_thr <= small_thr:
            pad = max(float(stat.get("std", 1.0)) * 0.25, 1e-3)
            small_thr -= pad
            large_thr += pad

        if value < small_thr:
            group = "small"
        elif value < large_thr:
            group = "medium"
        else:
            group = "large"

        denom = max(large_thr - small_thr, 1e-6)
        scale_prob = clamp((value - small_thr) / denom)
        return group, scale_prob, {
            "scale_mode": source,
            "scale_log_area": float(value),
            "scale_small_thr": small_thr,
            "scale_large_thr": large_thr,
        }

    def _absolute_group_from_area(self, area):
        small_thr = float(getattr(self.args, "scale_small_abs_thr", self.args.small_area_thr))
        large_thr = float(getattr(self.args, "scale_large_abs_thr", self.args.large_area_thr))
        if area < small_thr:
            return "small"
        if area < large_thr:
            return "medium"
        return "large"

    def _abs_scale_prob(self, area):
        small_thr = float(getattr(self.args, "scale_small_abs_thr", self.args.small_area_thr))
        large_thr = float(getattr(self.args, "scale_large_abs_thr", self.args.large_area_thr))
        if large_thr <= small_thr:
            return 0.0
        return clamp((float(area) - small_thr) / max(large_thr - small_thr, 1e-6))

    def _route_scale_level(self, score, scale_group):
        candidate_area = max(
            finite_float(score.get("scale_candidate_area", 0.0), 0.0),
            finite_float(score.get("area", 0.0), 0.0),
        )
        small_thr = float(getattr(self.args, "scale_small_abs_thr", self.args.small_area_thr))
        large_thr = float(getattr(self.args, "scale_large_abs_thr", self.args.large_area_thr))
        if scale_group == "small":
            if small_thr <= 0:
                return 0.0
            return clamp(candidate_area / max(small_thr, 1e-6))
        if scale_group == "medium":
            if large_thr <= small_thr:
                return clamp(score.get("scale_prob", 0.0))
            return clamp((candidate_area - small_thr) / max(large_thr - small_thr, 1e-6))
        if candidate_area <= large_thr:
            return 1.0
        extent_factor = max(float(self.args.large_max_area_ratio), 1.0001)
        extent = clamp(math.log(candidate_area / max(large_thr, 1e-6) + 1e-12) / math.log(extent_factor))
        return 1.0 + extent

    def _route_support_score(self, score):
        prompt_score = clamp(finite_float(score.get("prompt_consistency_score", 0.0), 0.0))
        p2seg_iou = clamp(finite_float(score.get("p2seg_iou", 0.0), 0.0))
        if score.get("feature_available", False):
            feature_support = clamp(finite_float(score.get("feature_box_iou", 0.0), 0.0))
        else:
            feature_support = 0.5
        return clamp(0.45 * prompt_score + 0.35 * p2seg_iou + 0.20 * feature_support)

    def _build_route_stats(self, scale_group, trust, score):
        if active_ablation(self.args) in {"no_scale_cues", "no_scale_conditioning"}:
            return {
                "route_profile": "uniform",
                "route_scale_level": 0.0,
                "route_support_score": 0.0,
                "route_density_penalty": 0.0,
                "route_ambiguity_penalty": 0.0,
                "route_effective_trust": float(trust),
                "route_direct_trust_thr": float(self.args.medium_direct_sam_trust_thr),
                "route_dynamic_fusion_cap": float(self.args.medium_max_fusion_weight),
                "route_min_prompt": float(self.args.medium_direct_sam_min_prompt),
                "route_min_p2seg_iou": float(self.args.medium_direct_sam_min_p2seg_iou),
                "route_min_feature_box_iou": float(self.args.medium_direct_sam_min_feature_box_iou),
            }
        route_scale_level = self._route_scale_level(score, scale_group)
        support_score = self._route_support_score(score)
        density_penalty = clamp(finite_float(score.get("local_point_density", 0.0), 0.0))
        ambiguity_penalty = scale_ambiguity_penalty(score.get("scale_ambiguity", "none"), scale_group)
        if is_reliability_free(self.args):
            support_score = 0.0
            density_penalty = 0.0
            ambiguity_penalty = 0.0

        if scale_group == "small":
            effective_trust = clamp(trust + 0.02 * support_score - 0.08 * ambiguity_penalty - 0.08 * density_penalty)
            return {
                "route_profile": "small_conservative",
                "route_scale_level": float(route_scale_level),
                "route_support_score": float(support_score),
                "route_density_penalty": float(density_penalty),
                "route_ambiguity_penalty": float(ambiguity_penalty),
                "route_effective_trust": float(effective_trust),
                "route_direct_trust_thr": float(self.args.small_high_trust_thr),
                "route_dynamic_fusion_cap": float(self.args.small_max_fusion_weight),
                "route_min_prompt": 0.0,
                "route_min_p2seg_iou": 0.0,
                "route_min_feature_box_iou": float(self.args.small_min_feature_box_iou),
            }

        if scale_group == "medium":
            progress = clamp(route_scale_level)
            effective_trust = clamp(trust + 0.08 * progress + 0.05 * support_score - 0.10 * ambiguity_penalty - 0.06 * density_penalty)
            direct_trust_thr = clamp(
                float(self.args.medium_direct_sam_trust_thr) - 0.08 * progress + 0.05 * ambiguity_penalty + 0.03 * density_penalty,
                float(self.args.low_trust_thr),
                0.95,
            )
            min_prompt = clamp(
                float(self.args.medium_direct_sam_min_prompt) - 0.10 * progress + 0.05 * ambiguity_penalty,
                0.35,
                0.95,
            )
            min_p2seg_iou = clamp(
                float(self.args.medium_direct_sam_min_p2seg_iou) - 0.08 * progress + 0.05 * ambiguity_penalty,
                0.15,
                0.95,
            )
            min_feature_box_iou = clamp(
                float(self.args.medium_direct_sam_min_feature_box_iou) - 0.05 * progress + 0.04 * ambiguity_penalty,
                0.05,
                0.95,
            )
            low_cap = max(float(self.args.small_max_fusion_weight), 0.40)
            cap_progress = clamp(progress + 0.20 * support_score - 0.20 * ambiguity_penalty - 0.10 * density_penalty)
            dynamic_fusion_cap = min(
                float(self.args.medium_max_fusion_weight),
                low_cap + (float(self.args.medium_max_fusion_weight) - low_cap) * cap_progress,
            )
            return {
                "route_profile": "medium_dynamic",
                "route_scale_level": float(route_scale_level),
                "route_support_score": float(support_score),
                "route_density_penalty": float(density_penalty),
                "route_ambiguity_penalty": float(ambiguity_penalty),
                "route_effective_trust": float(effective_trust),
                "route_direct_trust_thr": float(direct_trust_thr),
                "route_dynamic_fusion_cap": float(dynamic_fusion_cap),
                "route_min_prompt": float(min_prompt),
                "route_min_p2seg_iou": float(min_p2seg_iou),
                "route_min_feature_box_iou": float(min_feature_box_iou),
            }

        large_progress = clamp(route_scale_level - 1.0)
        effective_trust = clamp(trust + 0.06 * large_progress + 0.05 * support_score - 0.08 * ambiguity_penalty - 0.05 * density_penalty)
        direct_trust_thr = clamp(
            float(self.args.large_direct_sam_trust_thr) - 0.05 * large_progress + 0.04 * ambiguity_penalty + 0.03 * density_penalty,
            float(self.args.low_trust_thr),
            0.95,
        )
        min_prompt = clamp(
            float(self.args.large_direct_sam_min_prompt) - 0.05 * large_progress + 0.04 * ambiguity_penalty,
            0.30,
            0.90,
        )
        min_p2seg_iou = clamp(
            float(self.args.large_direct_sam_min_p2seg_iou) - 0.05 * large_progress + 0.04 * ambiguity_penalty,
            0.10,
            0.90,
        )
        dynamic_fusion_cap = clamp(
            0.60 + 0.25 * support_score + 0.10 * large_progress - 0.10 * ambiguity_penalty - 0.05 * density_penalty,
            0.45,
            1.0,
        )
        dynamic_fusion_cap = min(dynamic_fusion_cap, float(getattr(self.args, "large_route_fusion_cap_ceiling", 1.0)))
        return {
            "route_profile": "large_dynamic",
            "route_scale_level": float(route_scale_level),
            "route_support_score": float(support_score),
            "route_density_penalty": float(density_penalty),
            "route_ambiguity_penalty": float(ambiguity_penalty),
            "route_effective_trust": float(effective_trust),
            "route_direct_trust_thr": float(direct_trust_thr),
            "route_dynamic_fusion_cap": float(dynamic_fusion_cap),
            "route_min_prompt": float(min_prompt),
            "route_min_p2seg_iou": float(min_p2seg_iou),
            "route_min_feature_box_iou": float(getattr(self.args, "medium_direct_sam_min_feature_box_iou", 0.0)),
        }

    def _log_area(self, area, image_area):
        if area <= 0 or image_area <= 0:
            return 0.0
        return float(math.log(float(area) / float(image_area) + 1e-12))

    def _normalized_abs_log_area(self, area, small_thr, large_thr):
        area = max(finite_float(area, 0.0), 0.0)
        if area <= 0:
            return 0.0
        lo = math.log(max(float(small_thr), 1.0))
        hi = math.log(max(float(large_thr), float(small_thr) + 1.0, 1.0001))
        return clamp((math.log(area) - lo) / max(hi - lo, 1e-6))

    def _add_router_v2_features(self, score, p2seg_box):
        small_thr = finite_float(score.get("scale_small_thr"), getattr(self.args, "scale_small_abs_thr", self.args.small_area_thr))
        large_thr = finite_float(score.get("scale_large_thr"), getattr(self.args, "scale_large_abs_thr", self.args.large_area_thr))
        p2seg_area = box_area_xyxy(p2seg_box) if p2seg_box is not None else finite_float(score.get("area", 0.0), 0.0)
        score["p2seg_log_area_norm"] = self._normalized_abs_log_area(p2seg_area, small_thr, large_thr)
        score["candidate_log_area_norm"] = self._normalized_abs_log_area(score.get("scale_candidate_area", p2seg_area), small_thr, large_thr)
        score["safe_sam_log_area_norm"] = self._normalized_abs_log_area(score.get("scale_safe_sam_area", 0.0), small_thr, large_thr)
        score["safe_feature_log_area_norm"] = self._normalized_abs_log_area(score.get("scale_safe_feature_area", 0.0), small_thr, large_thr)
        if p2seg_box is None:
            score["p2seg_aspect_ratio"] = 1.0
            return
        w = max(float(p2seg_box[2] - p2seg_box[0]), 1e-6)
        h = max(float(p2seg_box[3] - p2seg_box[1]), 1e-6)
        score["p2seg_aspect_ratio"] = float(min(max(w / h, 0.05), 20.0))

    def _safe_sam_area(self, score, sam_box):
        if sam_box is None:
            return 0.0
        if score.get("same_class_hits", score.get("same_class_point_hits", 0)) > 0:
            return 0.0
        if float(score.get("sam_score", 0.0)) < float(self.args.gate_min_sam_score):
            return 0.0
        if float(score.get("p2seg_iou", 0.0)) < float(self.args.gate_min_p2seg_iou):
            return 0.0
        area_ratio = float(score.get("area_ratio", 1.0))
        if area_ratio < float(self.args.gate_min_area_ratio) or area_ratio > float(self.args.gate_max_area_ratio):
            return 0.0
        return box_area_xyxy(sam_box)

    def _safe_feature_area(self, score, feature_box):
        if feature_box is None or not score.get("feature_available", False):
            return 0.0
        if score.get("same_class_hits", score.get("same_class_point_hits", 0)) > 0:
            return 0.0
        if float(score.get("feature_consistency_score", 0.0)) < float(self.args.small_min_feature_score):
            return 0.0
        if float(score.get("feature_box_iou", 0.0)) < float(self.args.small_min_feature_box_iou):
            return 0.0
        return box_area_xyxy(feature_box)

    def _calibrated_group_from_area(self, area, image_area, score, sam_box, feature_box, relative_group, relative_meta):
        small_thr = float(getattr(self.args, "scale_small_abs_thr", self.args.small_area_thr))
        large_thr = float(getattr(self.args, "scale_large_abs_thr", self.args.large_area_thr))
        promote_thr = float(getattr(self.args, "scale_small_promote_area", small_thr * 1.5))
        base_area = max(float(area), 0.0)
        safe_sam_area = self._safe_sam_area(score, sam_box)
        safe_feature_area = self._safe_feature_area(score, feature_box)
        candidate_area = max(base_area, safe_sam_area, safe_feature_area)
        abs_group = self._absolute_group_from_area(base_area)
        candidate_abs_group = self._absolute_group_from_area(candidate_area)
        group = abs_group
        ambiguity = "none"
        reason = f"abs_{abs_group}"

        if abs_group == "small":
            if candidate_area >= large_thr:
                group = "large"
                ambiguity = "abs_small_but_credible_large"
                reason = "credible_candidate_promote_large"
            elif candidate_area >= promote_thr:
                group = "medium"
                ambiguity = "abs_small_but_credible_medium"
                reason = "credible_candidate_promote_medium"
            elif relative_group != "small":
                ambiguity = f"rel_{relative_group}_abs_small"
                reason = "absolute_small_blocks_relative_upscale"
        elif abs_group == "medium":
            if candidate_area >= large_thr:
                group = "large"
                ambiguity = "abs_medium_but_credible_large"
                reason = "credible_candidate_promote_large"
            elif relative_group == "small":
                ambiguity = "rel_small_abs_medium"
                reason = "absolute_medium_blocks_relative_small"
            elif relative_group == "large":
                ambiguity = "rel_large_abs_medium"
                reason = "absolute_medium_blocks_relative_large"
        else:
            if relative_group != "large":
                ambiguity = f"rel_{relative_group}_abs_large"
                reason = "absolute_large_blocks_relative_downscale"

        if abs(base_area - small_thr) <= small_thr * 0.10 or abs(base_area - large_thr) <= large_thr * 0.10:
            ambiguity = "boundary" if ambiguity == "none" else ambiguity

        meta = dict(relative_meta)
        meta.update({
            "scale_mode": "calibrated",
            "scale_abs_group": abs_group,
            "scale_relative_group": relative_group,
            "scale_candidate_abs_group": candidate_abs_group,
            "scale_ambiguity": ambiguity,
            "scale_head_reason": reason,
            "scale_log_area": self._log_area(base_area, image_area),
            "scale_small_thr": small_thr,
            "scale_large_thr": large_thr,
            "scale_relative_small_thr": float(relative_meta.get("scale_small_thr", 0.0)),
            "scale_relative_large_thr": float(relative_meta.get("scale_large_thr", 0.0)),
            "scale_candidate_area": float(candidate_area),
            "scale_safe_sam_area": float(safe_sam_area),
            "scale_safe_feature_area": float(safe_feature_area),
        })
        return group, self._abs_scale_prob(candidate_area), meta

    def _select_scale_stat(self, category_id):
        category_stat = self.scale_priors.get("class", {}).get(str(category_id))
        if category_stat and int(category_stat.get("count", 0)) >= int(self.args.scale_min_class_count):
            return category_stat, "class_adaptive"
        global_stat = self.scale_priors.get("global")
        if global_stat:
            return global_stat, "global_adaptive"
        return None, "none"

    def _absolute_scale_prob(self, area):
        small_thr = float(self.args.small_area_thr)
        large_thr = float(self.args.large_area_thr)
        if large_thr <= small_thr:
            return 0.0
        return clamp((float(area) - small_thr) / max(large_thr - small_thr, 1e-6))

    def local_point_density(self, point, other_anns, p2seg_box, current_category):
        if point is None:
            return 0.0, 0, 0
        area = max(box_area_xyxy(p2seg_box), 1.0)
        radius = max(math.sqrt(area) * float(self.args.local_density_radius_scale), 1.0)
        count = 0
        same = 0
        for ann in other_anns:
            other_point = point_from_ann(ann)
            if other_point is None:
                continue
            dist = math.hypot(float(other_point[0]) - float(point[0]), float(other_point[1]) - float(point[1]))
            if dist <= radius:
                count += 1
                if ann.get("category_id") == current_category:
                    same += 1
        norm = math.pi * radius * radius / max(area, 1.0)
        density = float(count) / max(norm, 1.0)
        return clamp(density / max(float(self.args.local_density_norm), 1e-6)), count, same

    def prompt_consistency(self, best, best_score, second_score, candidates, context, priors):
        if best is None or not candidates:
            return 0.0
        best_mask = best["mask"]
        ious = []
        for candidate in candidates:
            if candidate is best:
                continue
            other_score = score_candidate(candidate, context, self.args, priors)
            if other_score is None:
                continue
            ious.append(mask_iou(best_mask, candidate["mask"]))
        if not ious:
            margin = score_margin(best_score, second_score)
            return clamp(0.5 + (0.0 if margin is None else margin))
        return float(np.mean(ious))

    def prompt_consistency_from_scored(self, best, best_score, second_score, scored_candidates):
        if best is None or not scored_candidates:
            return 0.0
        ious = []
        for candidate, _ in scored_candidates:
            if candidate is best:
                continue
            ious.append(mask_iou(best["mask"], candidate["mask"]))
        if not ious:
            margin = score_margin(best_score, second_score)
            return clamp(0.5 + (0.0 if margin is None else margin))
        return float(np.mean(ious))

    def decode_final_box(self, candidate, score, p2seg_box, context):
        img_info = context["img_info"]
        ann = context["ann"]
        image_area = float(img_info["height"] * img_info["width"])
        sam_box = mask_to_xyxy(candidate["mask"])
        if sam_box is not None:
            sam_box = clip_box(sam_box, img_info["width"], img_info["height"])
        if p2seg_box is None and "bbox" in context["pred"]:
            p2seg_box = bbox_xywh_to_xyxy(context["pred"]["bbox"])
        if p2seg_box is not None:
            p2seg_box = clip_box(p2seg_box, img_info["width"], img_info["height"])

        feature_box = score.get("feature_box")
        if feature_box is not None:
            feature_box = clip_box(feature_box, img_info["width"], img_info["height"])

        base_area = box_area_xyxy(p2seg_box) if p2seg_box is not None else float(score.get("area", 0.0))
        scale_group, scale_prob, scale_meta = self.group_from_area(
            base_area, ann.get("category_id"), image_area, score, sam_box, feature_box)
        score.update(scale_meta)
        routing_scale_group = scale_group
        if active_ablation(self.args) in {"no_small_safety", "no_small_anchor"} and scale_group == "small":
            routing_scale_group = "medium"
        if active_ablation(self.args) in {"no_scale_cues", "no_scale_conditioning"}:
            routing_scale_group = "medium"
        score["routing_scale_group"] = routing_scale_group
        score["ablation"] = active_ablation(self.args)

        sam_box, boundary_info = calibrate_box_with_feature(
            sam_box,
            feature_box,
            score,
            score.get("scale_abs_group", scale_group),
            img_info,
            self.args,
        )
        score.update(boundary_info)

        density_score, density_count, same_density_count = self.local_point_density(
            context.get("point"),
            context.get("other_anns", []),
            p2seg_box,
            ann.get("category_id"),
        )
        score["local_point_density"] = density_score
        score["local_point_count"] = int(density_count)
        score["local_same_class_point_count"] = int(same_density_count)

        feature_score = float(score.get("feature_consistency_score", 0.0))
        scale_score = scale_consistency(score, p2seg_box, sam_box, context["point"], scale_group, self.args)
        if active_ablation(self.args) in {"no_scale_cues", "no_scale_conditioning"}:
            scale_prob = 0.5
            scale_score["scale_consistency_score"] = 1.0
        score.update(scale_score)
        sam_score = clamp(candidate.get("sam_score", 0.0))
        prompt_score = clamp(score.get("prompt_consistency_score", 0.0))
        geom_score = clamp(0.45 * score.get("p2seg_iou", 0.0) + 0.35 * scale_score["scale_consistency_score"] + 0.20 * prompt_score)
        density_penalty = clamp(1.0 - density_score)
        small_penalty = 0.0
        if routing_scale_group == "small":
            small_penalty = clamp((float(score.get("area_ratio", 1.0)) - 1.0) / max(float(self.args.small_max_area_ratio) - 1.0, 1e-6))
        trust = clamp(
            self.args.sam_trust_weight * sam_score
            + self.args.feature_trust_weight * feature_score
            + self.args.scale_trust_weight * scale_score["scale_consistency_score"]
            + self.args.geometry_trust_weight * geom_score
            + self.args.scale_prob_trust_weight * scale_prob
            + self.args.density_trust_weight * density_penalty
            - self.args.small_route_penalty_weight * small_penalty
        )
        if is_reliability_free(self.args):
            trust = 1.0
        route_stats = self._build_route_stats(routing_scale_group, trust, score)
        self._add_router_v2_features(score, p2seg_box)
        scale_safety_group = score.get("scale_abs_group", scale_group)
        if bypass_small_anchor(self.args):
            scale_safety_group = "medium"
        score["scale_safety_group"] = scale_safety_group
        v2_info = self._mlp_v2_release(score, route_stats, p2seg_box, sam_box, feature_box, img_info)
        if v2_info is not None:
            route_stats.update(v2_info)

        final_box = p2seg_box
        source = "p2seg"
        final_action = "fallback"
        fusion_weight = 0.0
        if active_ablation(self.args) == "uniform_fusion":
            if sam_box is not None and p2seg_box is not None:
                fusion_weight = float(self.args.uniform_fusion_weight)
                final_box = lerp_box_xyxy(p2seg_box, sam_box, fusion_weight)
                source = "fusion"
                final_action = "uniform_fusion"
            elif sam_box is not None:
                final_box = sam_box
                source = "sam2"
                final_action = "direct_sam"
        elif active_ablation(self.args) == "all_direct_sam":
            final_box = sam_box if sam_box is not None else p2seg_box
            source = "sam2" if sam_box is not None else "p2seg"
            final_action = "direct_sam" if sam_box is not None else "fallback"
            fusion_weight = 1.0 if sam_box is not None else 0.0
        elif is_sage_ablation(self.args):
            source, policy_reasons = sage_policy(self.args, score, sam_box, p2seg_box)
            final_box = p2seg_box if source == "p2seg" else sam_box
            final_action = "sage_p2seg_preserve" if source == "p2seg" else "sage_direct_sam"
            fusion_weight = 0.0 if source == "p2seg" else 1.0
            score["source_policy"] = active_ablation(self.args)
            score["source_policy_reasons"] = policy_reasons
        elif self.args.final_box_policy == "raw_sam_bbox_first":
            final_box, source, final_action, fusion_weight = self._decode_raw_sam_bbox_first(
                score,
                sam_box,
                p2seg_box,
                scale_group,
                feature_box,
                route_stats,
                img_info,
            )
        else:
            if self.args.disable_final_fusion:
                final_box = sam_box if sam_box is not None else p2seg_box
                source = "sam2" if sam_box is not None else "p2seg"
                final_action = "direct_sam" if sam_box is not None else "fallback"
                fusion_weight = 1.0 if sam_box is not None else 0.0
            elif routing_scale_group == "small":
                final_box, source, fusion_weight = self._decode_small_box(
                    trust, sam_box, p2seg_box, feature_box, feature_score, score, route_stats)
            elif routing_scale_group == "medium":
                final_box, source, fusion_weight = self._decode_medium_box(trust, sam_box, p2seg_box, score, route_stats)
            else:
                final_box, source, fusion_weight = self._decode_large_box(
                    trust, sam_box, p2seg_box, feature_box, feature_score, score, route_stats)
            if final_action == "fallback":
                if source == "sam2":
                    final_action = "direct_sam"
                elif source == "fusion":
                    final_action = "light_refine"
                elif source == "feature":
                    final_action = "feature_override"

        if final_box is None:
            final_box = sam_box
            source = "sam2" if sam_box is not None else source
            final_action = "direct_sam" if sam_box is not None else final_action
        final_box, source, fusion_weight = self._apply_small_safety_guard(
            final_box, source, fusion_weight, p2seg_box, sam_box, scale_safety_group, img_info)
        final_box = self._cap_small_box(final_box, p2seg_box, routing_scale_group, img_info)

        route_info = {
            "final_box": None if final_box is None else [float(v) for v in final_box],
            "final_source": source,
            "final_action": final_action,
            "sam2_trust_score": float(trust),
            "scale_prob": float(scale_prob),
            "fusion_weight": float(fusion_weight),
            "scale_safety_group": scale_safety_group,
            "sam_box": None if sam_box is None else [float(v) for v in sam_box],
            "local_point_density": float(density_score),
            "local_point_count": int(density_count),
            "local_same_class_point_count": int(same_density_count),
            "same_class_point_hits": int(score.get("same_class_hits", score.get("same_class_point_hits", 0))),
            "other_class_point_hits": int(score.get("other_class_hits", score.get("other_class_point_hits", 0))),
            "source_policy": score.get("source_policy", "legacy"),
            "source_policy_reasons": score.get("source_policy_reasons", []),
            **scale_meta,
            **scale_score,
            **route_stats,
        }
        mlp_info = self._mlp_override(score, route_info, p2seg_box, sam_box, feature_box, img_info)
        if mlp_info is not None:
            route_info.update(mlp_info)
        return route_info

    def _raw_sam_keep_ok(self, score, scale_group, route_stats):
        if score.get("sam_box") is None:
            return False
        trust = float(route_stats.get("route_effective_trust", score.get("sam2_trust_score", 0.0)))
        min_prompt = min(float(self.args.raw_sam_keep_min_prompt), float(route_stats.get("route_min_prompt", self.args.raw_sam_keep_min_prompt)))
        min_p2seg_iou = min(float(self.args.raw_sam_keep_min_p2seg_iou), float(route_stats.get("route_min_p2seg_iou", self.args.raw_sam_keep_min_p2seg_iou)))
        if trust < float(self.args.raw_sam_keep_min_trust):
            return False
        if float(score.get("prompt_consistency_score", 0.0)) < min_prompt:
            return False
        if float(score.get("p2seg_iou", 0.0)) < min_p2seg_iou:
            return False
        if float(score.get("scale_consistency_score", 0.0)) < float(self.args.gate_min_scale_score):
            return False
        if score.get("feature_available", False) and float(score.get("feature_consistency_score", 0.0)) < float(self.args.gate_min_feature_score):
            return False
        if scale_group == "small":
            if float(score.get("area_ratio", 1.0)) > float(self.args.small_max_area_ratio):
                return False
            if float(score.get("center_shift", 0.0)) > float(self.args.small_max_center_shift_px):
                return False
            if score.get("feature_available", False) and float(score.get("feature_box_iou", 0.0)) < float(self.args.small_min_feature_box_iou):
                return False
        return True

    def _raw_sam_force_fallback(self, score, scale_group):
        if score.get("sam_box") is None:
            return True
        if float(score.get("sam_score", 0.0)) < float(self.args.gate_min_sam_score):
            return True
        if float(score.get("p2seg_iou", 0.0)) < float(self.args.raw_sam_fallback_min_p2seg_iou):
            return True
        if scale_group == "small" and float(score.get("center_shift", 0.0)) > float(self.args.raw_sam_small_fallback_center_shift_px):
            return True
        return False

    def _light_refine_weight(self, score, scale_group):
        if scale_group == "small":
            max_weight = float(self.args.raw_sam_small_light_refine_max_weight)
        elif scale_group == "medium":
            max_weight = float(self.args.raw_sam_medium_light_refine_max_weight)
        else:
            max_weight = float(self.args.raw_sam_large_light_refine_max_weight)
        p2seg_disagree = clamp(1.0 - float(score.get("p2seg_iou", 0.0)))
        area_ratio = max(float(score.get("area_ratio", 1.0)), 1e-6)
        area_deviation = clamp(abs(math.log(area_ratio)) / math.log(max(float(self.args.gate_max_area_ratio), 1.0001)))
        severity = clamp(0.65 * p2seg_disagree + 0.35 * area_deviation)
        return max_weight * max(0.25, severity)

    def _decode_raw_sam_bbox_first(self, score, sam_box, p2seg_box, scale_group, feature_box, route_stats, img_info):
        score["sam_box"] = None if sam_box is None else [float(v) for v in sam_box]
        if sam_box is None:
            return p2seg_box, "p2seg", "fallback", 0.0
        if self._raw_sam_keep_ok(score, scale_group, route_stats):
            return sam_box, "sam2_raw", "keep_raw", 0.0
        if self._raw_sam_force_fallback(score, scale_group):
            return p2seg_box, "p2seg", "fallback", 0.0
        if p2seg_box is None:
            return sam_box, "sam2_raw", "keep_raw", 0.0
        weight = self._light_refine_weight(score, scale_group)
        final_box = lerp_box_xyxy(sam_box, p2seg_box, weight)
        final_box = clip_box(final_box, img_info["width"], img_info["height"])
        return final_box, "light_refine", "light_refine", weight

    def _mlp_feature_keys(self):
        if self.mlp is None:
            return ROUTER_FEATURE_KEYS
        return ROUTER_FEATURE_KEYS_BY_VERSION[int(self.mlp.get("router_head_version", 1))]

    def _mlp_hidden(self, score, route_info=None):
        route_info = route_info or {}
        keys = self._mlp_feature_keys()
        x = np.asarray([finite_float(score.get(key, route_info.get(key, 0.0))) for key in keys], dtype=np.float32)
        mean = np.asarray(self.mlp["mean"], dtype=np.float32)
        std = np.maximum(np.asarray(self.mlp["std"], dtype=np.float32), 1e-6)
        hidden_w = np.asarray(self.mlp["hidden_weight"], dtype=np.float32)
        hidden_b = np.asarray(self.mlp["hidden_bias"], dtype=np.float32)
        return np.maximum(0.0, ((x - mean) / std).dot(hidden_w.T) + hidden_b)

    def _mlp_v2_release(self, score, route_stats, p2seg_box, sam_box, feature_box, img_info):
        if self.mlp is None or int(self.mlp.get("router_head_version", 1)) != 2:
            return None
        scale_safety_group = score.get("scale_safety_group", score.get("scale_abs_group", score.get("scale_group", "unknown")))
        h = self._mlp_hidden(score)
        scale_logits = h.dot(np.asarray(self.mlp["scale_weight"], dtype=np.float32).T) + np.asarray(self.mlp["scale_bias"], dtype=np.float32)
        scale_probs = softmax(scale_logits)
        scale_idx = int(scale_probs.argmax())
        scale_conf = float(scale_probs[scale_idx])
        release_logit = float(h.dot(np.asarray(self.mlp["release_weight"], dtype=np.float32).reshape(-1)) + float(self.mlp["release_bias"]))
        fusion_delta = math.tanh(float(h.dot(np.asarray(self.mlp["fusion_delta_weight"], dtype=np.float32).reshape(-1)) + float(self.mlp["fusion_delta_bias"])))
        release_score = clamp(1.0 / (1.0 + math.exp(-release_logit)))
        info = {
            "router_head_version": 2,
            "router_mlp_used": False,
            "router_mlp_scale_conf": scale_conf,
            "router_mlp_route_conf": release_score,
            "perceived_scale_group": ROUTER_SCALE_NAMES.get(scale_idx, "unknown"),
            "router_release_score": release_score,
            "router_fusion_weight_delta": float(fusion_delta),
        }
        if scale_safety_group == "small":
            info["router_mlp_reject_reason"] = "small_safety"
            return info
        if score.get("scale_group") not in {"medium", "large"}:
            info["router_mlp_reject_reason"] = "non_releasable_scale"
            return info
        if scale_conf < float(self.args.router_mlp_min_scale_conf) or release_score < float(self.args.router_mlp_min_release_score):
            info["router_mlp_reject_reason"] = "low_conf"
            return info
        release_strength = release_score * min(max(scale_conf, 0.0), 1.0)
        direct_floor = float(self.args.low_trust_thr)
        route_stats["route_direct_trust_thr"] = float(max(direct_floor, route_stats["route_direct_trust_thr"] - self.args.router_mlp_release_trust_delta * release_strength))
        route_stats["route_min_prompt"] = float(max(0.30, route_stats["route_min_prompt"] - self.args.router_mlp_release_prompt_delta * release_strength))
        route_stats["route_min_p2seg_iou"] = float(max(0.10, route_stats["route_min_p2seg_iou"] - self.args.router_mlp_release_iou_delta * release_strength))
        route_stats["route_dynamic_fusion_cap"] = float(min(self.args.router_mlp_max_fusion_cap, route_stats["route_dynamic_fusion_cap"] + self.args.router_mlp_fusion_delta_scale * max(0.0, fusion_delta)))
        info.update({
            "router_mlp_used": True,
            "router_mlp_reject_reason": "",
            "route_direct_trust_thr": route_stats["route_direct_trust_thr"],
            "route_min_prompt": route_stats["route_min_prompt"],
            "route_min_p2seg_iou": route_stats["route_min_p2seg_iou"],
            "route_dynamic_fusion_cap": route_stats["route_dynamic_fusion_cap"],
        })
        return info

    def _mlp_override(self, score, route_info, p2seg_box, sam_box, feature_box, img_info):
        if self.mlp is None or int(self.mlp.get("router_head_version", 1)) != 1:
            return None
        h = self._mlp_hidden(score, route_info)
        scale_logits = h.dot(np.asarray(self.mlp["scale_weight"], dtype=np.float32).T) + np.asarray(self.mlp["scale_bias"], dtype=np.float32)
        route_logits = h.dot(np.asarray(self.mlp["route_weight"], dtype=np.float32).T) + np.asarray(self.mlp["route_bias"], dtype=np.float32)
        fusion_w = np.asarray(self.mlp["fusion_weight"], dtype=np.float32).reshape(-1)
        fusion_logit = float(h.dot(fusion_w) + float(self.mlp["fusion_bias"]))
        scale_probs = softmax(scale_logits)
        route_probs = softmax(route_logits)
        route_idx = int(route_probs.argmax())
        scale_idx = int(scale_probs.argmax())
        route_conf = float(route_probs[route_idx])
        scale_conf = float(scale_probs[scale_idx])
        if route_conf < float(self.args.router_mlp_min_route_conf) or scale_conf < float(self.args.router_mlp_min_scale_conf):
            return {
                "router_mlp_used": False,
                "router_mlp_route_conf": route_conf,
                "router_mlp_scale_conf": scale_conf,
            }
        source = ROUTER_ROUTE_NAMES.get(route_idx, "p2seg")
        if not self._mlp_safe(source, route_info, score):
            return {
                "router_mlp_used": False,
                "router_mlp_route_conf": route_conf,
                "router_mlp_scale_conf": scale_conf,
                "router_mlp_reject_reason": "unsafe_route",
            }
        fusion_weight = clamp(1.0 / (1.0 + math.exp(-fusion_logit)))
        if source == "feature" and feature_box is None:
            return {
                "router_mlp_used": False,
                "router_mlp_route_conf": route_conf,
                "router_mlp_scale_conf": scale_conf,
                "router_mlp_reject_reason": "missing_feature_box",
            }
        final_box = self._box_from_source(source, fusion_weight, p2seg_box, sam_box, feature_box, route_info.get("scale_group"), img_info)
        if final_box is None:
            return {
                "router_mlp_used": False,
                "router_mlp_route_conf": route_conf,
                "router_mlp_scale_conf": scale_conf,
                "router_mlp_reject_reason": "missing_box",
            }
        final_box, source, fusion_weight = self._apply_small_safety_guard(
            final_box,
            source,
            fusion_weight,
            p2seg_box,
            sam_box,
            route_info.get("scale_safety_group", route_info.get("scale_abs_group", route_info.get("scale_group"))),
            img_info,
        )
        return {
            "router_mlp_used": True,
            "router_mlp_route_conf": route_conf,
            "router_mlp_scale_conf": scale_conf,
            "router_mlp_scale_group": ROUTER_SCALE_NAMES.get(scale_idx, "unknown"),
            "final_source": source,
            "fusion_weight": float(fusion_weight if source == "fusion" else route_info.get("fusion_weight", 0.0)),
            "final_box": [float(v) for v in final_box],
        }

    def _mlp_safe(self, source, route_info, score):
        if source == "feature" and bool(getattr(self.args, "disable_feature_final_source", False)):
            return False
        if source == "p2seg":
            return True
        if route_info.get("scale_safety_group", score.get("scale_safety_group")) == "small":
            return False
        if score.get("same_class_hits", 0) > 0:
            return False
        scale_group = route_info.get("scale_group", score.get("scale_group"))
        if route_info.get("sam2_trust_score", score.get("sam2_trust_score", 0.0)) < float(self.args.low_trust_thr):
            return False
        if score.get("scale_consistency_score", 0.0) < float(self.args.gate_min_scale_score):
            return False
        if score.get("feature_available", False) and score.get("feature_consistency_score", 0.0) < float(self.args.gate_min_feature_score):
            return False
        if score.get("p2seg_iou", 0.0) < float(self.args.gate_min_p2seg_iou):
            return False
        if source == "feature":
            if not score.get("feature_available", False):
                return False
            if score.get("feature_consistency_score", 0.0) < float(self.args.feature_box_override_thr):
                return False
        if scale_group == "small":
            if source == "sam2":
                return False
            if score.get("area_ratio", 1.0) > float(self.args.small_max_area_ratio):
                return False
            if score.get("center_shift", 0.0) > float(self.args.small_max_center_shift_px):
                return False
            if score.get("feature_available", False) and score.get("feature_consistency_score", 0.0) < float(self.args.small_min_feature_score):
                return False
            if score.get("feature_available", False) and score.get("feature_box_iou", 0.0) < float(self.args.small_min_feature_box_iou):
                return False
        elif scale_group == "medium":
            if source == "sam2" and score.get("area_ratio", 1.0) > float(self.args.gate_max_area_ratio):
                return False
            if source == "fusion" and score.get("area_ratio", 1.0) > float(self.args.gate_max_area_ratio):
                return False
        else:
            if source == "sam2" and score.get("feature_available", False) and score.get("feature_consistency_score", 0.0) < float(self.args.gate_min_feature_score):
                return False
        return True

    def _box_from_source(self, source, fusion_weight, p2seg_box, sam_box, feature_box, scale_group, img_info):
        if source == "feature" and bool(getattr(self.args, "disable_feature_final_source", False)):
            source = "fusion" if sam_box is not None and p2seg_box is not None else ("sam2" if sam_box is not None else "p2seg")
        if source == "p2seg":
            final_box = p2seg_box
        elif source == "sam2":
            final_box = sam_box
        elif source == "feature":
            final_box = feature_box
        elif source == "fusion":
            final_box = lerp_box_xyxy(p2seg_box, sam_box, fusion_weight)
        else:
            final_box = None
        return self._cap_small_box(final_box, p2seg_box, scale_group, img_info)

    def _apply_small_safety_guard(self, final_box, source, fusion_weight, p2seg_box, sam_box, scale_safety_group, img_info):
        if bypass_small_anchor(self.args):
            return final_box, source, fusion_weight
        if final_box is None or p2seg_box is None or scale_safety_group != "small":
            return final_box, source, fusion_weight
        max_weight = min(float(self.args.small_safe_max_fusion_weight), float(self.args.small_max_fusion_weight))
        if source == "sam2":
            if sam_box is not None:
                source = "fusion"
                fusion_weight = max_weight
                final_box = lerp_box_xyxy(p2seg_box, sam_box, fusion_weight)
            else:
                source = "p2seg"
                fusion_weight = 0.0
                final_box = p2seg_box
        elif source == "fusion" and fusion_weight > max_weight:
            fusion_weight = max_weight
            final_box = lerp_box_xyxy(p2seg_box, sam_box, fusion_weight) if sam_box is not None else p2seg_box

        p2_area = max(box_area_xyxy(p2seg_box), 1.0)
        final_area = max(box_area_xyxy(final_box), 1.0)
        max_area = p2_area * float(self.args.small_safe_max_area_ratio)
        if final_area > max_area:
            shrink = math.sqrt(max_area / final_area)
            cx, cy = box_center_xyxy(final_box)
            w = max(1.0, float(final_box[2] - final_box[0]) * shrink)
            h = max(1.0, float(final_box[3] - final_box[1]) * shrink)
            final_box = clip_box([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], img_info["width"], img_info["height"])
        return final_box, source, fusion_weight

    def _decode_small_box(self, trust, sam_box, p2seg_box, feature_box, feature_score, score, route_stats=None):
        if sam_box is None or p2seg_box is None:
            return p2seg_box, "p2seg", 0.0
        effective_trust = trust if route_stats is None else float(route_stats.get("route_effective_trust", trust))
        dynamic_cap = float(self.args.small_max_fusion_weight if route_stats is None else route_stats.get("route_dynamic_fusion_cap", self.args.small_max_fusion_weight))
        strict_ok = (
            effective_trust >= float(self.args.small_high_trust_thr)
            and score.get("same_class_hits", 0) == 0
            and score.get("area_ratio", 1.0) <= float(self.args.small_max_area_ratio)
            and score.get("center_shift", 0.0) <= float(self.args.small_max_center_shift_px)
            and (not score.get("feature_available", False) or feature_score >= float(self.args.small_min_feature_score))
            and (not score.get("feature_available", False) or score.get("feature_box_iou", 0.0) >= float(self.args.small_min_feature_box_iou))
        )
        if is_reliability_free(self.args):
            strict_ok = sam_box is not None and p2seg_box is not None
        if strict_ok:
            weight = min(dynamic_cap, clamp((effective_trust - self.args.low_trust_thr) / max(self.args.high_trust_thr - self.args.low_trust_thr, 1e-6)))
            if (
                not bool(getattr(self.args, "disable_feature_final_source", False))
                and feature_box is not None
                and feature_score >= self.args.feature_box_override_thr
            ):
                return lerp_box_xyxy(p2seg_box, feature_box, weight), "feature", weight
            return lerp_box_xyxy(p2seg_box, sam_box, weight), "fusion", weight
        return p2seg_box, "p2seg", 0.0

    def _decode_medium_box(self, trust, sam_box, p2seg_box, score, route_stats=None):
        effective_trust = trust if route_stats is None else float(route_stats.get("route_effective_trust", trust))
        direct_trust_thr = float(self.args.medium_direct_sam_trust_thr if route_stats is None else route_stats.get("route_direct_trust_thr", self.args.medium_direct_sam_trust_thr))
        min_prompt = float(self.args.medium_direct_sam_min_prompt if route_stats is None else route_stats.get("route_min_prompt", self.args.medium_direct_sam_min_prompt))
        min_p2seg_iou = float(self.args.medium_direct_sam_min_p2seg_iou if route_stats is None else route_stats.get("route_min_p2seg_iou", self.args.medium_direct_sam_min_p2seg_iou))
        min_feature_box_iou = float(self.args.medium_direct_sam_min_feature_box_iou if route_stats is None else route_stats.get("route_min_feature_box_iou", self.args.medium_direct_sam_min_feature_box_iou))
        dynamic_cap = float(self.args.medium_max_fusion_weight if route_stats is None else route_stats.get("route_dynamic_fusion_cap", self.args.medium_max_fusion_weight))
        ambiguity = score.get("scale_ambiguity", "none")
        blocked_ambiguities = {"abs_small_but_credible_medium"}
        if not bool(getattr(self.args, "medium_allow_rel_small_abs_medium_direct_sam", False)):
            blocked_ambiguities.add("rel_small_abs_medium")
        direct_sam_ambiguity_ok = ambiguity not in blocked_ambiguities
        direct_sam_ok = (
            direct_sam_ambiguity_ok
            and float(score.get("prompt_consistency_score", 0.0)) >= min_prompt
            and float(score.get("p2seg_iou", 0.0)) >= min_p2seg_iou
            and (not score.get("feature_available", False) or float(score.get("feature_box_iou", 0.0)) >= min_feature_box_iou)
        )
        if is_reliability_free(self.args):
            direct_sam_ok = sam_box is not None
        if active_ablation(self.args) in {"no_direct_sam", "no_medium_large_direct"}:
            direct_sam_ok = False
        if effective_trust >= direct_trust_thr and sam_box is not None and direct_sam_ok:
            return sam_box, "sam2", 1.0
        if effective_trust >= self.args.low_trust_thr and p2seg_box is not None and sam_box is not None:
            low, high = float(self.args.low_trust_thr), float(self.args.high_trust_thr)
            weight = min(dynamic_cap, clamp((effective_trust - low) / max(high - low, 1e-6)))
            return lerp_box_xyxy(p2seg_box, sam_box, weight), "fusion", weight
        return p2seg_box, "p2seg", 0.0

    def _decode_large_box(self, trust, sam_box, p2seg_box, feature_box, feature_score, score, route_stats=None):
        effective_trust = trust if route_stats is None else float(route_stats.get("route_effective_trust", trust))
        direct_trust_thr = float(self.args.large_direct_sam_trust_thr if route_stats is None else route_stats.get("route_direct_trust_thr", self.args.large_direct_sam_trust_thr))
        min_prompt = float(self.args.large_direct_sam_min_prompt if route_stats is None else route_stats.get("route_min_prompt", self.args.large_direct_sam_min_prompt))
        min_p2seg_iou = float(self.args.large_direct_sam_min_p2seg_iou if route_stats is None else route_stats.get("route_min_p2seg_iou", self.args.large_direct_sam_min_p2seg_iou))
        dynamic_cap = float(1.0 if route_stats is None else route_stats.get("route_dynamic_fusion_cap", 1.0))
        if sam_box is not None and effective_trust >= self.args.low_trust_thr:
            if (
                not bool(getattr(self.args, "disable_feature_final_source", False))
                and feature_box is not None
                and feature_score >= self.args.feature_box_override_thr
                and bbox_iou_xyxy(feature_box, sam_box) < self.args.feature_sam_disagree_iou
            ):
                return lerp_box_xyxy(sam_box, feature_box, self.args.feature_box_fusion_weight), "feature", 1.0
            prompt_ok = float(score.get("prompt_consistency_score", 0.0)) >= min_prompt
            p2seg_ok = float(score.get("p2seg_iou", 0.0)) >= min_p2seg_iou
            if is_reliability_free(self.args):
                prompt_ok = True
                p2seg_ok = True
            if active_ablation(self.args) not in {"no_direct_sam", "no_medium_large_direct"} and effective_trust >= direct_trust_thr and prompt_ok and p2seg_ok:
                return sam_box, "sam2", 1.0
            if p2seg_box is not None:
                weight = min(dynamic_cap, clamp((effective_trust - self.args.low_trust_thr) / max(self.args.high_trust_thr - self.args.low_trust_thr, 1e-6)))
                return lerp_box_xyxy(p2seg_box, sam_box, weight), "fusion", weight
        return p2seg_box, "p2seg", 0.0

    def _cap_small_box(self, final_box, p2seg_box, scale_group, img_info):
        if final_box is None or p2seg_box is None or scale_group != "small":
            return final_box
        p2_area = max(box_area_xyxy(p2seg_box), 1.0)
        final_area = max(box_area_xyxy(final_box), 1.0)
        max_area = p2_area * float(self.args.small_final_max_area_ratio)
        if final_area <= max_area:
            return final_box
        shrink = math.sqrt(max_area / final_area)
        cx, cy = box_center_xyxy(final_box)
        w = max(1.0, float(final_box[2] - final_box[0]) * shrink)
        h = max(1.0, float(final_box[3] - final_box[1]) * shrink)
        return clip_box([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], img_info["width"], img_info["height"])


def find_image_path(img_root, file_name):
    file_path = Path(file_name)
    if file_path.is_absolute() and file_path.exists():
        return file_path
    root = Path(img_root)
    candidates = [
        root / file_name,
        root / "VOCdevkit" / file_name,
        root / Path(file_name).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def build_prompt_specs(point, negative_points, p2seg_box, p2seg_mask_prompt, width, height, args, semantic_box=None, semantic_box_score=0.0):
    prompt_modes = parse_prompt_modes(args.prompt_modes)
    box_ratios = parse_float_list(args.box_expand_ratios)
    specs = []

    point_coords = None
    point_labels = None
    if point is not None:
        coords = [[point[0], point[1]]]
        labels = [1]
        for neg_point, _ in negative_points:
            coords.append([neg_point[0], neg_point[1]])
            labels.append(0)
        point_coords = np.asarray(coords, dtype=np.float32)
        point_labels = np.asarray(labels, dtype=np.int32)

    boxes = []
    if p2seg_box is not None:
        for ratio in box_ratios:
            boxes.append((ratio, expand_box(p2seg_box, ratio, width, height)))

    if "point" in prompt_modes and point_coords is not None:
        specs.append({"name": "point", "point_coords": point_coords, "point_labels": point_labels, "box": None, "mask_input": None, "multimask": True})
        jitter_radius = max(0, int(getattr(args, "point_jitter_radius", 0)))
        if jitter_radius > 0:
            for dx, dy in ((-jitter_radius, 0), (jitter_radius, 0), (0, -jitter_radius), (0, jitter_radius)):
                jittered = point_coords.copy()
                jittered[0, 0] = min(max(jittered[0, 0] + dx, 0.0), float(width - 1))
                jittered[0, 1] = min(max(jittered[0, 1] + dy, 0.0), float(height - 1))
                specs.append({"name": f"point_jitter_{dx}_{dy}", "point_coords": jittered, "point_labels": point_labels, "box": None, "mask_input": None, "multimask": True})
    if "box" in prompt_modes:
        for ratio, box in boxes:
            specs.append({"name": f"box_{ratio:g}", "point_coords": None, "point_labels": None, "box": box, "mask_input": None, "multimask": True})
    if "point_box" in prompt_modes and point_coords is not None:
        for ratio, box in boxes:
            specs.append({"name": f"point_box_{ratio:g}", "point_coords": point_coords, "point_labels": point_labels, "box": box, "mask_input": None, "multimask": True})
    if semantic_box is not None and point_coords is not None:
        specs.append({"name": "semantic_point_box", "point_coords": point_coords, "point_labels": point_labels, "box": semantic_box, "mask_input": None, "multimask": True, "semantic_box_score": semantic_box_score})
        if p2seg_mask_prompt is not None:
            specs.append({"name": "semantic_point_box_mask", "point_coords": point_coords, "point_labels": point_labels, "box": semantic_box, "mask_input": p2seg_mask_prompt, "multimask": False, "semantic_box_score": semantic_box_score})
    if "mask" in prompt_modes and p2seg_mask_prompt is not None:
        specs.append({"name": "mask", "point_coords": None, "point_labels": None, "box": None, "mask_input": p2seg_mask_prompt, "multimask": False})
    if "point_mask" in prompt_modes and point_coords is not None and p2seg_mask_prompt is not None:
        specs.append({"name": "point_mask", "point_coords": point_coords, "point_labels": point_labels, "box": None, "mask_input": p2seg_mask_prompt, "multimask": False})
    if "point_box_mask" in prompt_modes and point_coords is not None and p2seg_mask_prompt is not None and boxes:
        specs.append({"name": "point_box_mask", "point_coords": point_coords, "point_labels": point_labels, "box": boxes[0][1], "mask_input": p2seg_mask_prompt, "multimask": False})
    return specs


def predict_candidates(predictor, specs):
    candidates = []
    for spec in specs:
        masks, ious, _ = predictor.predict(
            point_coords=spec["point_coords"],
            point_labels=spec["point_labels"],
            box=spec["box"],
            mask_input=spec["mask_input"],
            multimask_output=spec["multimask"],
        )
        for idx, mask in enumerate(masks):
            candidates.append({
                "mask": mask.astype(bool),
                "sam_score": float(ious[idx]) if idx < len(ious) else 0.0,
                "prompt": spec["name"],
                "semantic_box_score": float(spec.get("semantic_box_score", 0.0)),
            })
    return candidates


def candidate_centrality(scored_candidates, topk, resolution):
    """Return prompt-ensemble support without using boxes or GT labels."""
    if not scored_candidates:
        return []
    topk = max(1, int(topk))
    resolution = max(16, int(resolution))
    pooled = np.stack([
        resize_bool_mask(candidate["mask"], (resolution, resolution)).reshape(-1).astype(np.float32)
        for candidate, _ in scored_candidates
    ])
    areas = pooled.sum(axis=1, keepdims=True)
    intersection = pooled @ pooled.T
    union = areas + areas.T - intersection
    overlaps = np.divide(intersection, np.maximum(union, 1e-6), out=np.zeros_like(intersection), where=union > 0)
    np.fill_diagonal(overlaps, 0.0)
    top = np.partition(overlaps, -min(topk, len(scored_candidates) - 1), axis=1)[:, -min(topk, len(scored_candidates) - 1):]
    return [float(value) for value in top.mean(axis=1)]


def normalize_candidate_values(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-8:
        return np.full_like(values, 0.5)
    return (values - low) / (high - low)


def rank_scored_candidates(scored_candidates, args):
    if not scored_candidates:
        return scored_candidates
    centrality = candidate_centrality(scored_candidates, args.candidate_centrality_topk, args.candidate_centrality_resolution)
    for (_, score), value in zip(scored_candidates, centrality):
        score["candidate_centrality"] = value
    if args.candidate_selection_policy == "centrality":
        return sorted(scored_candidates, key=lambda item: item[1]["candidate_centrality"], reverse=True)
    if args.candidate_selection_policy == "centrality_sam":
        normalized_centrality = normalize_candidate_values(centrality)
        normalized_sam = normalize_candidate_values([score.get("sam_score", 0.0) for _, score in scored_candidates])
        sam_weight = clamp(float(args.candidate_sam_weight))
        for (_, score), centrality_value, sam_value in zip(scored_candidates, normalized_centrality, normalized_sam):
            score["candidate_combined_score"] = float((1.0 - sam_weight) * centrality_value + sam_weight * sam_value)
        return sorted(scored_candidates, key=lambda item: item[1]["candidate_combined_score"], reverse=True)
    return sorted(scored_candidates, key=lambda item: item[1]["total"], reverse=True)


def normalize_feature_map(feat):
    feat = np.asarray(feat, dtype=np.float32)
    if feat.ndim == 4:
        feat = feat[0]
    if feat.ndim != 3:
        return None
    if feat.shape[-1] <= 32:
        pass
    elif feat.shape[0] in (1, 3):
        feat = np.transpose(feat, (1, 2, 0))
    elif feat.shape[-1] in (1, 3):
        pass
    elif feat.shape[0] <= 32 and feat.shape[-1] > 32:
        feat = np.transpose(feat, (1, 2, 0))
    elif feat.shape[1] == feat.shape[2] and feat.shape[0] != feat.shape[1]:
        feat = np.transpose(feat, (1, 2, 0))
    norm = np.linalg.norm(feat, axis=2, keepdims=True)
    return feat / np.maximum(norm, 1e-6)


def extract_predictor_feature_map(predictor):
    candidates = []
    for attr in ("features", "_features", "image_embed", "_image_embed"):
        if hasattr(predictor, attr):
            candidates.append(getattr(predictor, attr))
    if hasattr(predictor, "_features") and isinstance(predictor._features, dict):
        candidates.extend(predictor._features.values())
    if hasattr(predictor, "features") and isinstance(predictor.features, dict):
        candidates.extend(predictor.features.values())

    for value in candidates:
        if isinstance(value, dict):
            candidates.extend(value.values())
            continue
        if isinstance(value, (list, tuple)):
            candidates.extend(value)
            continue
        try:
            if hasattr(value, "detach"):
                value = value.detach().float().cpu().numpy()
            elif hasattr(value, "cpu") and hasattr(value, "numpy"):
                value = value.cpu().numpy()
            feat = normalize_feature_map(value)
        except Exception:
            feat = None
        if feat is not None:
            return feat
    return None


def build_rgb_feature_map(image, args):
    stride = max(1, int(args.rgb_feature_stride))
    arr = image.astype(np.float32) / 255.0
    img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    small = img.resize((max(1, image.shape[1] // stride), max(1, image.shape[0] // stride)), Image.BILINEAR)
    base = np.asarray(small, dtype=np.float32) / 255.0

    gray = 0.299 * base[:, :, 0] + 0.587 * base[:, :, 1] + 0.114 * base[:, :, 2]
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    yy, xx = np.mgrid[0:base.shape[0], 0:base.shape[1]].astype(np.float32)
    xx = xx / max(base.shape[1] - 1, 1)
    yy = yy / max(base.shape[0] - 1, 1)

    feat = np.dstack([base, gray[:, :, None], gx[:, :, None], gy[:, :, None], xx[:, :, None], yy[:, :, None]])
    return normalize_feature_map(feat)


def build_feature_map(image, predictor, args):
    if args.feature_source in ("auto", "sam2"):
        feat = extract_predictor_feature_map(predictor)
        if feat is not None:
            return feat, "sam2"
        if args.feature_source == "sam2":
            return None, "none"
    if args.feature_source in ("auto", "rgb"):
        return build_rgb_feature_map(image, args), "rgb"
    return None, "none"


def local_point_prototype(feature_map, point, image_width, image_height, radius):
    if feature_map is None or point is None:
        return None, None
    feat_h, feat_w = feature_map.shape[:2]
    feat_point = scale_point(point, image_width, image_height, feat_w, feat_h)
    if feat_point is None:
        return None, None
    px, py = feat_point
    radius = max(0, int(radius))
    x1, x2 = max(0, px - radius), min(feat_w, px + radius + 1)
    y1, y2 = max(0, py - radius), min(feat_h, py + radius + 1)
    patch = feature_map[y1:y2, x1:x2]
    if patch.size == 0:
        return None, feat_point
    proto = patch.reshape(-1, patch.shape[-1]).mean(axis=0)
    proto = proto / max(float(np.linalg.norm(proto)), 1e-6)
    return proto.astype(np.float32), feat_point


def similarity_map_from_point(feature_map, point, image_width, image_height, args):
    proto, feat_point = local_point_prototype(
        feature_map, point, image_width, image_height, args.feature_proto_radius)
    if proto is None:
        return None, None
    sim = np.tensordot(feature_map, proto, axes=([2], [0]))
    sim = (sim + 1.0) * 0.5
    return np.clip(sim, 0.0, 1.0).astype(np.float32), feat_point


def feature_consistency(candidate_mask, p2seg_mask, p2seg_box, sam_box, point, sim_map, feat_point, img_info, args):
    empty = {
        "feature_available": False,
        "feature_consistency_score": 0.0,
        "feature_inside_mean": 0.0,
        "feature_ring_mean": 0.0,
        "feature_support_ratio": 0.0,
        "feature_box_iou": 0.0,
        "feature_box": None,
        "feature_component_area": 0.0,
        "feature_threshold": 0.0,
    }
    if sim_map is None or feat_point is None:
        return empty

    feat_h, feat_w = sim_map.shape
    candidate_feat = resize_bool_mask(candidate_mask, (feat_h, feat_w))
    p2seg_feat = resize_bool_mask(p2seg_mask, (feat_h, feat_w))
    search_feat = np.logical_or(candidate_feat, p2seg_feat)
    if not np.any(search_feat):
        return empty

    values = sim_map[search_feat]
    adaptive_th = float(values.mean() + args.feature_threshold_std * max(values.std(), 1e-6))
    threshold = max(float(args.feature_min_threshold), min(float(args.feature_max_threshold), adaptive_th))
    high_sim = np.logical_and(sim_map >= threshold, search_feat)
    component = component_from_point(high_sim, feat_point)
    component_area = float(component.sum())
    feature_box = feature_box_to_image_box(
        mask_to_xyxy(component),
        feat_w,
        feat_h,
        img_info["width"],
        img_info["height"],
    )

    inside_vals = sim_map[candidate_feat]
    inside_mean = float(inside_vals.mean()) if inside_vals.size else 0.0
    union_image_box = union_box_xyxy(p2seg_box, sam_box)
    union_feat_box = image_box_to_feature_box(union_image_box, img_info["width"], img_info["height"], feat_w, feat_h)
    ring_mask = np.logical_and(
        expand_box_mask(union_feat_box, (feat_h, feat_w), args.feature_ring_expand_ratio),
        np.logical_not(candidate_feat),
    )
    ring_vals = sim_map[ring_mask]
    ring_mean = float(ring_vals.mean()) if ring_vals.size else float(values.mean())
    contrast = clamp((inside_mean - ring_mean + 0.25) / 0.5)

    support_ratio = 0.0
    if candidate_feat.sum() > 0:
        support_ratio = float(np.logical_and(component, candidate_feat).sum()) / float(candidate_feat.sum())
    feature_box_iou = bbox_iou_xyxy(feature_box, sam_box)
    support_score = clamp(support_ratio / max(float(args.feature_support_target), 1e-6))
    box_score = clamp(feature_box_iou / max(float(args.feature_box_iou_target), 1e-6))
    feature_score = clamp(0.50 * contrast + 0.30 * support_score + 0.20 * box_score)

    return {
        "feature_available": True,
        "feature_consistency_score": float(feature_score),
        "feature_inside_mean": inside_mean,
        "feature_ring_mean": ring_mean,
        "feature_support_ratio": float(support_ratio),
        "feature_box_iou": float(feature_box_iou),
        "feature_box": None if feature_box is None else [float(v) for v in feature_box],
        "feature_component_area": component_area,
        "feature_threshold": threshold,
    }


def scale_group_from_area(area, args):
    if area < float(args.small_area_thr):
        return "small"
    if area < float(args.large_area_thr):
        return "medium"
    return "large"


def scale_consistency(score, p2seg_box, sam_box, point, scale_group, args):
    area_ratio = float(score.get("area_ratio", 0.0))
    ratio_score = math.exp(-abs(math.log(max(area_ratio, 1e-6))))
    if scale_group == "small":
        max_ratio = float(args.small_max_area_ratio)
        center_norm = float(args.small_center_shift_norm)
    elif scale_group == "medium":
        max_ratio = float(args.medium_max_area_ratio)
        center_norm = float(args.medium_center_shift_norm)
    else:
        max_ratio = float(args.large_max_area_ratio)
        center_norm = float(args.large_center_shift_norm)

    if area_ratio > max_ratio:
        ratio_score *= max(0.0, max_ratio / max(area_ratio, 1e-6))

    center_score = 1.0
    center_shift = 0.0
    center = box_center_xyxy(sam_box)
    if center is not None and point is not None and p2seg_box is not None:
        diag = math.sqrt(max(box_area_xyxy(p2seg_box), 1.0))
        center_shift = float(np.linalg.norm(center - np.asarray(point, dtype=np.float32)))
        center_score = math.exp(-center_shift / max(center_norm * diag, 1e-6))

    point_hit_score = 1.0
    if score.get("same_class_hits", 0) > 0:
        point_hit_score *= 0.0
    if score.get("other_class_hits", 0) > 0:
        point_hit_score *= max(0.0, 1.0 - 0.25 * float(score["other_class_hits"]))

    return {
        "scale_consistency_score": float(clamp(0.55 * ratio_score + 0.30 * center_score + 0.15 * point_hit_score)),
        "center_shift": center_shift,
        "scale_group": scale_group,
        "scale_max_area_ratio": max_ratio,
    }


def score_candidate(candidate, context, args, priors):
    mask = candidate["mask"]
    p2seg_mask = context["p2seg_mask"]
    point = context["point"]
    other_anns = context["other_anns"]
    ann = context["ann"]
    img_info = context["img_info"]
    image_preds = context["image_preds"]
    semantic_ranker = context.get("semantic_ranker")

    area = float(mask.sum())
    if area <= 0:
        return None

    own_inside = point_inside_mask(mask, point)
    if args.require_own_point and point is not None and not own_inside:
        return None

    same_class_hits = 0
    other_class_hits = 0
    for other_ann in other_anns:
        other_point = point_from_ann(other_ann)
        if other_point is None or not point_inside_mask(mask, other_point):
            continue
        if other_ann.get("category_id") == ann.get("category_id"):
            same_class_hits += 1
        else:
            other_class_hits += 1

    if args.reject_other_same_class_point and same_class_hits > 0:
        return None

    p2seg_area = max(float(p2seg_mask.sum()), 1.0)
    area_ratio = area / p2seg_area
    if area_ratio < args.min_area_ratio or area_ratio > args.max_area_ratio:
        return None

    p2seg_iou = mask_iou(mask, p2seg_mask)
    if not is_reliability_free(args) and p2seg_iou < args.min_p2seg_iou:
        return None

    sam_score = max(0.0, min(1.0, float(candidate["sam_score"])))
    semantic_score = 0.0 if semantic_ranker is None else semantic_ranker.score(context["image"], mask, ann["category_id"])
    point_score = 1.0 if own_inside else 0.0
    point_score -= args.same_class_point_penalty * same_class_hits
    point_score -= args.other_class_point_penalty * other_class_hits
    area_change_score = math.exp(-abs(math.log(area_ratio + 1e-12)))

    image_area = float(img_info["height"] * img_info["width"])
    cls_area_score = class_area_score(area, image_area, ann["category_id"], priors)
    rel_score = relative_size_score(area, ann["category_id"], image_preds, ann["id"], priors)
    sam_box = mask_to_xyxy(mask)
    feature_info = feature_consistency(
        mask,
        p2seg_mask,
        context.get("p2seg_box"),
        sam_box,
        point,
        context.get("sim_map"),
        context.get("feat_point"),
        img_info,
        args,
    )

    if is_reliability_free(args):
        total = args.sam_score_weight * sam_score + args.point_weight * point_score
    elif active_ablation(args) == "no_scale_cues":
        total = (
            args.sam_score_weight * sam_score
            + args.p2seg_iou_weight * p2seg_iou
            + args.point_weight * point_score
            + args.area_change_weight * area_change_score
            + args.feature_weight * feature_info["feature_consistency_score"]
        )
    else:
        total = (
            args.sam_score_weight * sam_score
            + args.p2seg_iou_weight * p2seg_iou
            + args.point_weight * point_score
            + args.area_change_weight * area_change_score
            + args.class_area_weight * cls_area_score
            + args.relative_size_weight * rel_score
            + args.feature_weight * feature_info["feature_consistency_score"]
        )
    if semantic_ranker is not None:
        total += float(args.semantic_rank_weight) * semantic_score
    quality = max(0.0, min(1.0, 0.35 * sam_score + 0.35 * p2seg_iou + 0.20 * max(0.0, point_score) + 0.10 * area_change_score))
    if args.class_area_weight > 0:
        quality *= 0.8 + 0.2 * cls_area_score
    if args.relative_size_weight > 0 and rel_score > 0:
        quality *= 0.85 + 0.15 * rel_score

    return {
        "sam_score": sam_score,
        "total": float(total),
        "total_score": float(total),
        "quality": float(quality),
        "area": area,
        "area_ratio": area_ratio,
        "p2seg_iou": p2seg_iou,
        "point_score": point_score,
        "same_class_hits": same_class_hits,
        "other_class_hits": other_class_hits,
        "same_class_point_hits": same_class_hits,
        "other_class_point_hits": other_class_hits,
        "class_area_score": cls_area_score,
        "relative_size_score": rel_score,
        "semantic_candidate_ranking": uses_semantic_candidate_ranking(args),
        "semantic_score": semantic_score,
        "semantic_box_score": float(candidate.get("semantic_box_score", 0.0)),
        "semantic_box_used": candidate.get("prompt", "").startswith("semantic_"),
        **feature_info,
    }


def pass_replace_gate(candidate, score, second_score, args):
    if args.final_box_policy == "raw_sam_bbox_first":
        action = score.get("final_action", "fallback")
        return action != "fallback", "route_fallback" if action == "fallback" else action
    if not args.enable_replace_gate:
        return True, "disabled"
    if is_sage_ablation(args):
        source = score.get("final_source")
        if source == "p2seg":
            return True, "ablation_{}_p2seg".format(active_ablation(args))
        return True, "ablation_{}_sam2".format(active_ablation(args))
    if is_reliability_free(args):
        if score.get("final_source") == "p2seg":
            return False, "route_p2seg"
        return True, "ablation_{}".format(active_ablation(args))
    score["gate_score_margin_thr"] = gate_margin_threshold(score, args)
    if score.get("final_source") == "p2seg":
        return False, "route_p2seg"
    if float(candidate.get("sam_score", 0.0)) < args.gate_min_sam_score:
        return False, "low_sam_score"
    if score["p2seg_iou"] < args.gate_min_p2seg_iou:
        return False, "low_p2seg_iou"
    if score["area_ratio"] < args.gate_min_area_ratio:
        return False, "small_area_ratio"
    if score["area_ratio"] > args.gate_max_area_ratio:
        return False, "large_area_ratio"
    if score["same_class_hits"] > args.gate_max_same_class_hits:
        return False, "same_class_point_hit"
    if score["other_class_hits"] > args.gate_max_other_class_hits:
        return False, "other_class_point_hit"
    if args.enable_scale_aware_gate:
        if score.get("sam2_trust_score", 0.0) < args.gate_min_trust_score:
            return False, "low_trust_score"
        if score.get("scale_consistency_score", 0.0) < args.gate_min_scale_score:
            return False, "low_scale_score"
        if score.get("feature_available", False) and score.get("feature_consistency_score", 0.0) < args.gate_min_feature_score:
            return False, "low_feature_score"
        if score.get("routing_scale_group", score.get("scale_group")) == "small":
            if score["area_ratio"] > args.small_max_area_ratio:
                return False, "small_large_area_ratio"
            if score.get("feature_available", False) and score.get("feature_box_iou", 0.0) < args.small_min_feature_box_iou:
                return False, "small_low_feature_support"
            if score.get("center_shift", 0.0) > args.small_max_center_shift_px:
                return False, "small_large_center_shift"
    if second_score is not None:
        margin = score["total"] - second_score["total"]
        if margin < score["gate_score_margin_thr"]:
            scale_group = score.get("routing_scale_group", score.get("scale_group", "unknown"))
            if scale_group in ROUTER_SCALE_LABELS:
                return False, f"{scale_group}_score_margin"
            return False, "score_margin"
    return True, "pass"


def score_margin(best_score, second_score):
    if best_score is None or second_score is None:
        return None
    return float(best_score["total"] - second_score["total"])


def make_refine_info(selected, gate_reason, candidate, score, second_score, feature_source):
    final_box = score.get("final_box")
    final_bbox = None if final_box is None or not selected else bbox_xyxy_to_xywh(final_box)
    return {
        "selected": bool(selected),
        "gate_reason": gate_reason,
        "prompt": candidate.get("prompt"),
        "sam_score": float(candidate.get("sam_score", 0.0)),
        "total_score": float(score.get("total_score", score.get("total", 0.0))),
        "score_margin": score.get("score_margin", score_margin(score, second_score)),
        "sam2_trust_score": float(score.get("sam2_trust_score", 0.0)),
        "scale_prob": float(score.get("scale_prob", 0.0)),
        "feature_consistency_score": float(score.get("feature_consistency_score", 0.0)),
        "scale_consistency_score": float(score.get("scale_consistency_score", 0.0)),
        "prompt_consistency_score": float(score.get("prompt_consistency_score", 0.0)),
        "feature_source": feature_source,
        "final_bbox": final_bbox,
        "final_box_xyxy": final_box if selected else None,
        "final_source": score.get("final_source", "unknown"),
        "final_action": score.get("final_action", "unknown"),
        "fusion_weight": float(score.get("fusion_weight", 0.0)),
        "sam_box": score.get("sam_box"),
        "feature_box": score.get("feature_box"),
        "scale_group": score.get("scale_group", "unknown"),
        "routing_scale_group": score.get("routing_scale_group", score.get("scale_group", "unknown")),
        "ablation": score.get("ablation", "full"),
        "scale_safety_group": score.get("scale_safety_group", score.get("scale_abs_group", "unknown")),
        "center_shift": float(score.get("center_shift", 0.0)),
        "area_ratio": float(score.get("area_ratio", 0.0)),
        "p2seg_iou": float(score.get("p2seg_iou", 0.0)),
        "feature_support_ratio": float(score.get("feature_support_ratio", 0.0)),
        "feature_box_iou": float(score.get("feature_box_iou", 0.0)),
        "feature_inside_mean": float(score.get("feature_inside_mean", 0.0)),
        "feature_ring_mean": float(score.get("feature_ring_mean", 0.0)),
        "feature_threshold": float(score.get("feature_threshold", 0.0)),
        "local_point_density": float(score.get("local_point_density", 0.0)),
        "local_point_count": int(score.get("local_point_count", 0)),
        "local_same_class_point_count": int(score.get("local_same_class_point_count", 0)),
        "scale_mode": score.get("scale_mode", "unknown"),
        "scale_abs_group": score.get("scale_abs_group", "unknown"),
        "scale_relative_group": score.get("scale_relative_group", "unknown"),
        "scale_candidate_abs_group": score.get("scale_candidate_abs_group", "unknown"),
        "scale_ambiguity": score.get("scale_ambiguity", "unknown"),
        "scale_head_reason": score.get("scale_head_reason", "unknown"),
        "scale_log_area": float(score.get("scale_log_area", 0.0)),
        "scale_small_thr": float(score.get("scale_small_thr", 0.0)),
        "scale_large_thr": float(score.get("scale_large_thr", 0.0)),
        "scale_relative_small_thr": float(score.get("scale_relative_small_thr", 0.0)),
        "scale_relative_large_thr": float(score.get("scale_relative_large_thr", 0.0)),
        "scale_candidate_area": float(score.get("scale_candidate_area", 0.0)),
        "scale_safe_sam_area": float(score.get("scale_safe_sam_area", 0.0)),
        "scale_safe_feature_area": float(score.get("scale_safe_feature_area", 0.0)),
        "p2seg_log_area_norm": float(score.get("p2seg_log_area_norm", 0.0)),
        "candidate_log_area_norm": float(score.get("candidate_log_area_norm", 0.0)),
        "safe_sam_log_area_norm": float(score.get("safe_sam_log_area_norm", 0.0)),
        "safe_feature_log_area_norm": float(score.get("safe_feature_log_area_norm", 0.0)),
        "p2seg_aspect_ratio": float(score.get("p2seg_aspect_ratio", 1.0)),
        "router_head_version": int(score.get("router_head_version", 1)),
        "router_mlp_used": bool(score.get("router_mlp_used", False)),
        "router_mlp_route_conf": float(score.get("router_mlp_route_conf", 0.0)),
        "router_mlp_scale_conf": float(score.get("router_mlp_scale_conf", 0.0)),
        "router_mlp_scale_group": score.get("router_mlp_scale_group", "unknown"),
        "router_mlp_reject_reason": score.get("router_mlp_reject_reason", ""),
        "perceived_scale_group": score.get("perceived_scale_group", "unknown"),
        "router_release_score": float(score.get("router_release_score", 0.0)),
        "router_fusion_weight_delta": float(score.get("router_fusion_weight_delta", 0.0)),
        "route_profile": score.get("route_profile", "unknown"),
        "route_scale_level": float(score.get("route_scale_level", 0.0)),
        "route_support_score": float(score.get("route_support_score", 0.0)),
        "route_density_penalty": float(score.get("route_density_penalty", 0.0)),
        "route_ambiguity_penalty": float(score.get("route_ambiguity_penalty", 0.0)),
        "route_effective_trust": float(score.get("route_effective_trust", 0.0)),
        "route_direct_trust_thr": float(score.get("route_direct_trust_thr", 0.0)),
        "route_dynamic_fusion_cap": float(score.get("route_dynamic_fusion_cap", 0.0)),
        "route_min_prompt": float(score.get("route_min_prompt", 0.0)),
        "route_min_p2seg_iou": float(score.get("route_min_p2seg_iou", 0.0)),
        "route_min_feature_box_iou": float(score.get("route_min_feature_box_iou", 0.0)),
        "source_policy": score.get("source_policy", "legacy"),
        "source_policy_reasons": score.get("source_policy_reasons", []),
        "class_area_score": float(score.get("class_area_score", 0.0)),
        "relative_size_score": float(score.get("relative_size_score", 0.0)),
        "semantic_candidate_ranking": bool(score.get("semantic_candidate_ranking", False)),
        "semantic_box_score": float(score.get("semantic_box_score", 0.0)),
        "semantic_box_used": bool(score.get("semantic_box_used", False)),
        "same_class_point_hits": int(score.get("same_class_hits", 0)),
        "other_class_point_hits": int(score.get("other_class_hits", 0)),
        "consensus_member_count": int(score.get("consensus_member_count", 0)),
        "consensus_weighted_iou": float(score.get("consensus_weighted_iou", 0.0)),
        "consensus_expansion": float(score.get("consensus_expansion", 0.0)),
        "boundary_calibration_applied": bool(score.get("boundary_calibration_applied", False)),
        "boundary_calibration_weight": float(score.get("boundary_calibration_weight", 0.0)),
        "raw_sam_box": score.get("raw_sam_box"),
        "boundary_calibrated_box": score.get("boundary_calibrated_box"),
    }


def overlay_debug(image, p2seg_mask, refined_mask, path):
    arr = image.astype(np.float32).copy()
    red = np.array([255, 0, 0], dtype=np.float32)
    green = np.array([0, 255, 0], dtype=np.float32)
    arr[p2seg_mask] = arr[p2seg_mask] * 0.55 + red * 0.45
    arr[refined_mask] = arr[refined_mask] * 0.55 + green * 0.45
    both = np.logical_and(p2seg_mask, refined_mask)
    arr[both] = arr[both] * 0.45 + np.array([255, 255, 0], dtype=np.float32) * 0.55
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def load_sam2_predictor(args):
    if args.sam2_root:
        sys.path.insert(0, str(Path(args.sam2_root).resolve()))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_model = build_sam2(args.sam2_config, args.sam2_ckpt, device=args.device)
    return SAM2ImagePredictor(sam2_model)


def refine_results(args):
    dataset = load_json(args.ann)
    results = load_json(args.result)
    semantic_boxes = load_semantic_boxes(args.semantic_box_json)
    images_by_id = {img["id"]: img for img in dataset["images"]}
    anns_by_id = {ann["id"]: ann for ann in dataset["annotations"]}
    anns_by_image = defaultdict(list)
    for ann in dataset["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    results_by_image = defaultdict(list)
    for pred in results:
        img = images_by_id.get(pred.get("image_id"))
        if img is not None:
            pred["_p2seg_area"] = mask_area_from_pred(pred, img)
        results_by_image[pred["image_id"]].append(pred)

    priors = load_priors(args.prior_json)
    if args.use_online_priors:
        online_priors = build_online_priors(results, images_by_id, args.min_relative_pair_count)
        if args.prior_json:
            online_priors["class_area"].update(priors.get("class_area", {}))
            online_priors["relative_area"].update(priors.get("relative_area", {}))
            online_priors["scale"] = priors.get("scale", {})
        priors = online_priors
    if args.use_adaptive_scale_priors:
        scale_priors = build_scale_priors(
            results,
            images_by_id,
            args.scale_small_quantile,
            args.scale_large_quantile,
        )
        if args.prior_json and priors.get("scale"):
            scale_priors["class"].update(priors.get("scale", {}).get("class", {}))
            if priors.get("scale", {}).get("global"):
                scale_priors["global"] = priors["scale"]["global"]
        priors["scale"] = scale_priors
    if args.save_prior_json:
        priors_to_save = {k: v for k, v in priors.items() if k != "area_by_ann_id"}
        if "scale" in priors_to_save:
            priors_to_save["scale"] = {k: v for k, v in priors_to_save["scale"].items() if k != "area_by_ann_id"}
        dump_json(priors_to_save, args.save_prior_json)

    selected_image_ids = list(results_by_image.keys())
    if args.image_ids:
        selected = {int(v) for v in args.image_ids.split(",") if v.strip()}
        selected_image_ids = [image_id for image_id in selected_image_ids if image_id in selected]
    if args.max_images > 0:
        selected_image_ids = selected_image_ids[: args.max_images]

    predictor = load_sam2_predictor(args)
    semantic_ranker = SemanticProposalRanker(args.semantic_head_checkpoint, args.device) if args.semantic_head_checkpoint else None
    router = ScaleAwareRouter(args, priors)
    mask_utils = get_mask_utils()
    refined = copy.deepcopy(results)
    refined_by_ann_id = {pred.get("ann_id"): pred for pred in refined if pred.get("ann_id") is not None}

    processed = 0
    updated = 0
    fallback = 0
    missing = 0
    gated = 0
    gate_reasons = defaultdict(int)
    final_sources = defaultdict(int)
    feature_sources = defaultdict(int)
    scale_groups = defaultdict(int)
    selected_scale_groups = defaultdict(int)
    selected_image_set = set(selected_image_ids)
    candidate_dump = []

    image_iter = progress_iter(
        enumerate(selected_image_ids, 1),
        total=len(selected_image_ids),
        desc="SAM2 refine",
        disable=args.no_progress,
    )
    for image_idx, image_id in image_iter:
        img_info = images_by_id[image_id]
        image_path = find_image_path(args.img_root, img_info["file_name"])
        if not image_path.exists():
            print(f"[WARN] missing image: {image_path}")
            missing += len(results_by_image[image_id])
            continue

        image = load_rgb(image_path)
        predictor.set_image(image)
        feature_map, feature_source = build_feature_map(image, predictor, args)
        feature_sources[feature_source] += 1
        image_preds = results_by_image[image_id]
        image_anns = anns_by_image.get(image_id, [])

        for pred in image_preds:
            if args.max_instances > 0 and processed >= args.max_instances:
                break
            ann_id = pred.get("ann_id")
            ann = anns_by_id.get(ann_id)
            target = refined_by_ann_id.get(ann_id)
            if ann is None or target is None:
                missing += 1
                continue

            point = point_from_ann(ann)
            point_for_negative = [point[0], point[1], ann_id] if point is not None else None
            other_anns = [a for a in image_anns if a["id"] != ann_id]
            negative_points = nearest_negative_points(point_for_negative, other_anns, anns_by_id, args.negative_points, args.max_negative_points)

            p2seg_mask = decode_segm(pred["segmentation"], img_info["height"], img_info["width"])
            p2seg_box = mask_to_xyxy(p2seg_mask)
            if p2seg_box is None and "bbox" in pred:
                p2seg_box = bbox_xywh_to_xyxy(pred["bbox"])
            if p2seg_box is not None:
                p2seg_box = clip_box(p2seg_box, img_info["width"], img_info["height"])
            p2seg_mask_prompt = make_mask_prompt(p2seg_mask)
            sim_map, feat_point = similarity_map_from_point(feature_map, point, img_info["width"], img_info["height"], args)

            semantic_box_info = semantic_boxes.get(ann_id)
            semantic_box = None
            semantic_box_score = 0.0
            if semantic_box_info is not None and semantic_box_info["score"] >= float(args.semantic_box_min_score):
                semantic_box = clip_box(semantic_box_info["box"], img_info["width"], img_info["height"])
                semantic_box_score = semantic_box_info["score"]
            specs = build_prompt_specs(
                point,
                negative_points,
                p2seg_box,
                p2seg_mask_prompt,
                img_info["width"],
                img_info["height"],
                args,
                semantic_box=semantic_box,
                semantic_box_score=semantic_box_score,
            )
            candidates = predict_candidates(predictor, specs)
            context = {
                "ann": ann,
                "pred": pred,
                "img_info": img_info,
                "point": point,
                "other_anns": other_anns,
                "p2seg_mask": p2seg_mask,
                "p2seg_box": p2seg_box,
                "image_preds": image_preds,
                "sim_map": sim_map,
                "feat_point": feat_point,
                "image": image,
                "semantic_ranker": semantic_ranker,
                "semantic_box": semantic_box,
                "semantic_box_score": semantic_box_score,
            }

            best = None
            best_score = None
            second_score = None
            scored_candidates = []
            for candidate in candidates:
                candidate_score = score_candidate(candidate, context, args, priors)
                if candidate_score is None:
                    continue
                candidate_score["consensus_member_count"] = int(candidate.get("consensus_member_count", 0))
                candidate_score["consensus_weighted_iou"] = float(candidate.get("consensus_weighted_iou", 0.0))
                candidate_score["consensus_expansion"] = float(candidate.get("consensus_expansion", 0.0))
                scored_candidates.append((candidate, candidate_score))

            consensus_candidate = mine_prompt_consensus(scored_candidates, point, args)
            if consensus_candidate is not None:
                consensus_score = score_candidate(consensus_candidate, context, args, priors)
                if consensus_score is not None:
                    consensus_score["consensus_member_count"] = int(consensus_candidate["consensus_member_count"])
                    consensus_score["consensus_weighted_iou"] = float(consensus_candidate["consensus_weighted_iou"])
                    consensus_score["consensus_expansion"] = float(consensus_candidate["consensus_expansion"])
                    scored_candidates.append((consensus_candidate, consensus_score))

            if scored_candidates:
                scored_candidates.sort(key=lambda item: item[1]["total"], reverse=True)
                color_candidate = color_refine_candidate(scored_candidates[0][0], context.get("image"), point, args)
                if color_candidate is not None:
                    color_score = score_candidate(color_candidate, context, args, priors)
                    if color_score is not None:
                        color_score["color_refine_mask_iou"] = float(color_candidate["color_refine_mask_iou"])
                        color_score["color_refine_area_ratio"] = float(color_candidate["color_refine_area_ratio"])
                        scored_candidates.append((color_candidate, color_score))

            if scored_candidates:
                scored_candidates = rank_scored_candidates(scored_candidates, args)
                best, best_score = scored_candidates[0]
                second_score = scored_candidates[1][1] if len(scored_candidates) > 1 else None

            if args.debug_candidate_json and scored_candidates:
                candidate_dump.append({
                    "ann_id": int(ann_id),
                    "image_id": int(image_id),
                    "candidates": [
                        {
                            "bbox": [float(value) for value in bbox_xyxy_to_xywh(mask_to_xyxy(candidate["mask"]))],
                            "prompt": candidate.get("prompt", ""),
                            "total_score": float(score.get("total", 0.0)),
                            "sam_score": float(score.get("sam_score", 0.0)),
                            "p2seg_iou": float(score.get("p2seg_iou", 0.0)),
                            "feature_score": float(score.get("feature_consistency_score", 0.0)),
                            "candidate_centrality": float(score.get("candidate_centrality", 0.0)),
                        }
                        for candidate, score in scored_candidates
                        if mask_to_xyxy(candidate["mask"]) is not None
                    ],
                })

            if best is None:
                fallback += 1
                target["sam2_refine"] = {
                    "selected": False,
                    "gate_reason": "no_candidate",
                    "feature_source": feature_source,
                    "final_source": "p2seg",
                    "final_action": "fallback",
                    "final_bbox": None,
                    "scale_group": "unknown",
                    "scale_prob": 0.0,
                    "sam2_trust_score": 0.0,
                    "feature_consistency_score": 0.0,
                    "scale_consistency_score": 0.0,
                    "prompt_consistency_score": 0.0,
                    "fusion_weight": 0.0,
                    "scale_abs_group": "unknown",
                    "scale_relative_group": "unknown",
                    "scale_candidate_abs_group": "unknown",
                    "scale_ambiguity": "no_candidate",
                    "scale_head_reason": "no_candidate",
                }
            else:
                best_score["prompt_consistency_score"] = router.prompt_consistency_from_scored(
                    best, best_score, second_score, scored_candidates)
                best_score["score_margin"] = score_margin(best_score, second_score)
                best_score.update(router.decode_final_box(best, best_score, p2seg_box, context))
                scale_groups[best_score.get("scale_group", "unknown")] += 1
                gate_ok, gate_reason = pass_replace_gate(best, best_score, second_score, args)
                if not gate_ok:
                    gated += 1
                    gate_reasons[gate_reason] += 1
                    target["sam2_refine"] = make_refine_info(False, gate_reason, best, best_score, second_score, feature_source)
                    processed += 1
                    continue

                refined_mask = p2seg_mask if best_score.get("final_source") == "p2seg" else best["mask"]
                rle = encode_mask(refined_mask)
                bbox = best_score.get("final_box")
                if bbox is None:
                    bbox = bbox_xywh_to_xyxy(mask_utils.toBbox(rle).tolist())
                bbox_xywh = bbox_xyxy_to_xywh(bbox)
                target["segmentation"] = rle
                target["bbox"] = [float(v) for v in bbox_xywh]
                orig_score = float(pred.get("score", 1.0))
                target["score"] = float(max(0.0, min(1.0, 0.5 * orig_score + 0.5 * best_score["quality"])))
                target["sam2_refine"] = make_refine_info(True, gate_reason, best, best_score, second_score, feature_source)
                target["sam2_refine"]["final_bbox"] = [float(v) for v in bbox_xywh]
                final_sources[best_score.get("final_source", "unknown")] += 1
                selected_scale_groups[best_score.get("scale_group", "unknown")] += 1
                updated += 1
                if args.debug_dir and updated <= args.debug_count:
                    debug_name = f"img{image_id}_ann{ann_id}_{best['prompt']}.jpg"
                    overlay_debug(image, p2seg_mask, refined_mask, Path(args.debug_dir) / debug_name)

            processed += 1

        if args.max_instances > 0 and processed >= args.max_instances:
            break
        if args.log_interval > 0 and image_idx % args.log_interval == 0:
            print(f"[INFO] images={image_idx}/{len(selected_image_ids)} processed={processed} updated={updated} fallback={fallback} gated={gated} sources={dict(sorted(final_sources.items()))}")

    for pred in refined:
        pred.pop("_p2seg_area", None)
    dump_json(refined, args.out)
    if args.debug_candidate_json:
        dump_json(candidate_dump, args.debug_candidate_json)
    print(f"processed={processed}, updated={updated}, fallback={fallback}, gated={gated}, missing={missing}, output={args.out}")
    if gate_reasons:
        print("gate_reasons=" + json.dumps(dict(sorted(gate_reasons.items())), sort_keys=True))
    if final_sources:
        print("final_sources=" + json.dumps(dict(sorted(final_sources.items())), sort_keys=True))
    if scale_groups:
        print("scale_groups=" + json.dumps(dict(sorted(scale_groups.items())), sort_keys=True))
    if selected_scale_groups:
        print("selected_scale_groups=" + json.dumps(dict(sorted(selected_scale_groups.items())), sort_keys=True))
    if feature_sources:
        print("feature_sources=" + json.dumps(dict(sorted(feature_sources.items())), sort_keys=True))
    if args.max_images > 0 or args.max_instances > 0:
        print("[WARN] max-images/max-instances was used; unprocessed entries were copied unchanged.")
    if selected_image_set and len(selected_image_set) < len(results_by_image):
        print(f"[INFO] selected_images={len(selected_image_set)} total_result_images={len(results_by_image)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Refine P2Seg/P2MNet COCO-format mask results with SAM2 prompts.")
    parser.add_argument("--ann", required=True, help="Original COCO-format annotation file.")
    parser.add_argument("--result", required=True, help="P2Seg/P2MNet segm result json with ann_id.")
    parser.add_argument("--img-root", required=True, help="Image root used with image file_name in annotation.")
    parser.add_argument("--out", required=True, help="Output refined segm result json.")
    parser.add_argument("--sam2-root", default="", help="SAM2 repository root; used when sam2 is not installed as a package.")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml", help="SAM2 config name passed to build_sam2.")
    parser.add_argument("--sam2-ckpt", required=True, help="SAM2 checkpoint path.")
    parser.add_argument("--device", default="cuda", help="Torch device for SAM2.")
    parser.add_argument("--prompt-modes", default="point,box,point_box,mask,point_mask,point_box_mask")
    parser.add_argument("--box-expand-ratios", default="0,0.10,0.20,-0.10")
    parser.add_argument("--negative-points", choices=["none", "same-class", "all"], default="all")
    parser.add_argument("--max-negative-points", type=int, default=8)
    parser.add_argument("--require-own-point", action="store_true", default=True)
    parser.add_argument("--allow-missing-own-point", dest="require_own_point", action="store_false")
    parser.add_argument("--reject-other-same-class-point", action="store_true", default=True)
    parser.add_argument("--allow-other-same-class-point", dest="reject_other_same_class_point", action="store_false")
    parser.add_argument("--min-p2seg-iou", type=float, default=0.03)
    parser.add_argument("--min-area-ratio", type=float, default=0.10)
    parser.add_argument("--max-area-ratio", type=float, default=10.0)
    parser.add_argument("--sam-score-weight", type=float, default=1.0)
    parser.add_argument("--p2seg-iou-weight", type=float, default=1.0)
    parser.add_argument("--point-weight", type=float, default=2.0)
    parser.add_argument("--area-change-weight", type=float, default=0.5)
    parser.add_argument("--class-area-weight", type=float, default=0.5)
    parser.add_argument("--relative-size-weight", type=float, default=0.3)
    parser.add_argument("--feature-weight", type=float, default=0.6, help="Weight of point-feature consistency in candidate ranking.")
    parser.add_argument("--prior-json", default="", help="Optional class/relative area prior json.")
    parser.add_argument("--use-online-priors", action="store_true", help="Build class and relative size priors from the input P2Seg result.")
    parser.add_argument("--save-prior-json", default="", help="Optional path to save priors built from current result.")
    parser.add_argument("--min-relative-pair-count", type=int, default=20)
    parser.add_argument("--same-class-point-penalty", type=float, default=2.0)
    parser.add_argument("--other-class-point-penalty", type=float, default=0.5)
    parser.add_argument("--enable-replace-gate", action="store_true", default=True, help="Only write SAM2 masks that pass conservative confidence checks.")
    parser.add_argument("--disable-replace-gate", dest="enable_replace_gate", action="store_false")
    parser.add_argument("--gate-min-sam-score", type=float, default=0.75)
    parser.add_argument("--gate-min-p2seg-iou", type=float, default=0.20)
    parser.add_argument("--gate-min-area-ratio", type=float, default=0.25)
    parser.add_argument("--gate-max-area-ratio", type=float, default=4.0)
    parser.add_argument("--gate-min-score-margin", type=float, default=0.05)
    parser.add_argument("--small-gate-min-score-margin", type=float, default=0.05)
    parser.add_argument("--medium-gate-min-score-margin", type=float, default=0.02)
    parser.add_argument("--large-gate-min-score-margin", type=float, default=0.01)
    parser.add_argument("--gate-max-same-class-hits", type=int, default=0)
    parser.add_argument("--gate-max-other-class-hits", type=int, default=999)
    parser.add_argument("--enable-scale-aware-gate", action="store_true", default=True, help="Use trust/feature/scale checks before writing SAM2 refined outputs.")
    parser.add_argument("--disable-scale-aware-gate", dest="enable_scale_aware_gate", action="store_false")
    parser.add_argument("--gate-min-trust-score", type=float, default=0.45)
    parser.add_argument("--gate-min-scale-score", type=float, default=0.30)
    parser.add_argument("--gate-min-feature-score", type=float, default=0.20)
    parser.add_argument("--small-min-feature-box-iou", type=float, default=0.05)
    parser.add_argument("--small-max-center-shift-px", type=float, default=48.0)
    parser.add_argument("--feature-source", choices=["auto", "sam2", "rgb", "none"], default="auto")
    parser.add_argument("--rgb-feature-stride", type=int, default=4)
    parser.add_argument("--feature-proto-radius", type=int, default=1)
    parser.add_argument("--feature-threshold-std", type=float, default=0.25)
    parser.add_argument("--feature-min-threshold", type=float, default=0.55)
    parser.add_argument("--feature-max-threshold", type=float, default=0.85)
    parser.add_argument("--feature-ring-expand-ratio", type=float, default=0.50)
    parser.add_argument("--feature-support-target", type=float, default=0.35)
    parser.add_argument("--feature-box-iou-target", type=float, default=0.35)
    parser.add_argument("--semantic-candidate-ranking", action="store_true", help="For all_direct_sam, rank SAM2 candidates with semantic and geometric cues while keeping the final source as SAM2.")
    parser.add_argument("--semantic-head-checkpoint", default="", help="Optional frozen-ResNet semantic MIL head trained from point-labelled P2Seg crops.")
    parser.add_argument("--semantic-rank-weight", type=float, default=1.0)
    parser.add_argument("--semantic-box-json", default="", help="Optional frozen category-localizer boxes keyed by annotation id; used only as additional SAM2 box prompts.")
    parser.add_argument("--semantic-box-min-score", type=float, default=0.0, help="Minimum frozen-localizer score required before adding its point-plus-box SAM2 prompts.")
    parser.add_argument("--candidate-selection-policy", choices=["score", "centrality", "centrality_sam"], default="score", help="Rank SAM2 prompt candidates by the legacy score, cross-prompt mask centrality, or centrality combined with SAM2 quality.")
    parser.add_argument("--candidate-centrality-topk", type=int, default=6, help="Number of most-overlapping prompt masks used for centrality ranking.")
    parser.add_argument("--candidate-centrality-resolution", type=int, default=64, help="Square mask resolution used for vectorized centrality IoU.")
    parser.add_argument("--candidate-sam-weight", type=float, default=0.30, help="Normalized SAM2 quality weight for centrality_sam candidate ranking.")
    parser.add_argument("--enable-color-boundary-refine", action="store_true", help="Add a point-anchored GrabCut boundary candidate after SAM2 selection.")
    parser.add_argument("--color-refine-roi-expand", type=float, default=0.15)
    parser.add_argument("--color-refine-point-radius", type=int, default=2)
    parser.add_argument("--color-refine-iterations", type=int, default=2)
    parser.add_argument("--color-refine-min-mask-iou", type=float, default=0.60)
    parser.add_argument("--color-refine-min-area-ratio", type=float, default=0.50)
    parser.add_argument("--color-refine-max-area-ratio", type=float, default=1.50)
    parser.add_argument("--color-refine-score-bonus", type=float, default=0.01)
    parser.add_argument("--point-jitter-radius", type=int, default=0, help="Generate four bounded positive-point perturbation prompts around each annotated point.")
    parser.add_argument("--enable-prompt-consensus", action="store_true", default=False, help="Aggregate compatible multi-prompt SAM2 masks before candidate routing.")
    parser.add_argument("--disable-prompt-consensus", dest="enable_prompt_consensus", action="store_false")
    parser.add_argument("--consensus-topk", type=int, default=3, help="Maximum compatible prompt candidates used by consensus mining.")
    parser.add_argument("--consensus-min-iou", type=float, default=0.45, help="Minimum anchor IoU for a prompt candidate to join consensus mining.")
    parser.add_argument("--consensus-min-containment", type=float, default=0.80, help="Minimum smaller-mask containment for scale-complementary prompt mining.")
    parser.add_argument("--consensus-min-support", type=float, default=0.55, help="Weighted prompt support required for a consensus mask pixel.")
    parser.add_argument("--consensus-temperature", type=float, default=0.20, help="Softmax temperature for prompt-consensus candidate weights.")
    parser.add_argument("--consensus-max-expansion", type=float, default=1.50, help="Maximum consensus-mask area relative to its top candidate.")
    parser.add_argument("--enable-boundary-calibration", action="store_true", default=False, help="Conservatively calibrate medium/large SAM2 boxes with point-connected feature boundaries.")
    parser.add_argument("--disable-boundary-calibration", dest="enable_boundary_calibration", action="store_false")
    parser.add_argument("--boundary-calibration-min-feature-score", type=float, default=0.70)
    parser.add_argument("--boundary-calibration-min-box-iou", type=float, default=0.30)
    parser.add_argument("--boundary-calibration-max-area-ratio", type=float, default=2.0)
    parser.add_argument("--boundary-calibration-min-weight", type=float, default=0.08)
    parser.add_argument("--boundary-calibration-max-weight", type=float, default=0.35)
    parser.add_argument("--use-adaptive-scale-priors", action="store_true", default=True, help="Estimate small/medium/large from dataset log area priors.")
    parser.add_argument("--disable-adaptive-scale-priors", dest="use_adaptive_scale_priors", action="store_false")
    parser.add_argument("--scale-head-mode", choices=["calibrated", "relative", "absolute"], default="calibrated")
    parser.add_argument("--scale-small-abs-thr", type=float, default=1024.0)
    parser.add_argument("--scale-large-abs-thr", type=float, default=9216.0)
    parser.add_argument("--scale-small-promote-area", type=float, default=1536.0)
    parser.add_argument("--scale-small-quantile", type=float, default=33.0)
    parser.add_argument("--scale-large-quantile", type=float, default=67.0)
    parser.add_argument("--scale-min-class-count", type=int, default=30)
    parser.add_argument("--small-area-thr", type=float, default=1024.0)
    parser.add_argument("--large-area-thr", type=float, default=9216.0)
    parser.add_argument("--small-max-area-ratio", type=float, default=2.0)
    parser.add_argument("--medium-max-area-ratio", type=float, default=4.0)
    parser.add_argument("--large-max-area-ratio", type=float, default=8.0)
    parser.add_argument("--small-center-shift-norm", type=float, default=1.0)
    parser.add_argument("--medium-center-shift-norm", type=float, default=1.5)
    parser.add_argument("--large-center-shift-norm", type=float, default=2.5)
    parser.add_argument("--sam-trust-weight", type=float, default=0.25)
    parser.add_argument("--feature-trust-weight", type=float, default=0.30)
    parser.add_argument("--scale-trust-weight", type=float, default=0.30)
    parser.add_argument("--geometry-trust-weight", type=float, default=0.15)
    parser.add_argument("--scale-prob-trust-weight", type=float, default=0.10)
    parser.add_argument("--density-trust-weight", type=float, default=0.05)
    parser.add_argument("--small-route-penalty-weight", type=float, default=0.20)
    parser.add_argument("--low-trust-thr", type=float, default=0.45)
    parser.add_argument("--high-trust-thr", type=float, default=0.70)
    parser.add_argument("--small-high-trust-thr", type=float, default=0.72)
    parser.add_argument("--small-min-feature-score", type=float, default=0.35)
    parser.add_argument("--small-max-fusion-weight", type=float, default=0.35)
    parser.add_argument("--medium-direct-sam-trust-thr", type=float, default=0.80)
    parser.add_argument("--medium-direct-sam-min-prompt", type=float, default=0.60)
    parser.add_argument("--medium-direct-sam-min-p2seg-iou", type=float, default=0.40)
    parser.add_argument("--medium-direct-sam-min-feature-box-iou", type=float, default=0.20)
    parser.add_argument("--medium-allow-rel-small-abs-medium-direct-sam", action="store_true", help="Allow rel_small_abs_medium samples to use direct SAM2 in medium routing while keeping other ambiguity blocks unchanged.")
    parser.add_argument("--medium-max-fusion-weight", type=float, default=0.80)
    parser.add_argument("--large-direct-sam-trust-thr", type=float, default=0.82)
    parser.add_argument("--large-direct-sam-min-prompt", type=float, default=0.55)
    parser.add_argument("--large-direct-sam-min-p2seg-iou", type=float, default=0.30)
    parser.add_argument("--router-mlp-json", default="", help="Optional V1 router MLP JSON exported by train_scale_router_mlp.py.")
    parser.add_argument("--router-mlp-min-route-conf", type=float, default=0.70)
    parser.add_argument("--router-mlp-min-scale-conf", type=float, default=0.60)
    parser.add_argument("--router-mlp-min-release-score", type=float, default=0.60)
    parser.add_argument("--router-mlp-release-trust-delta", type=float, default=0.08)
    parser.add_argument("--router-mlp-release-prompt-delta", type=float, default=0.08)
    parser.add_argument("--router-mlp-release-iou-delta", type=float, default=0.08)
    parser.add_argument("--router-mlp-fusion-delta-scale", type=float, default=0.15)
    parser.add_argument("--router-mlp-max-fusion-cap", type=float, default=0.85)
    parser.add_argument("--final-box-policy", choices=["legacy", "raw_sam_bbox_first"], default="legacy", help="How to turn the SAM candidate into the final training box.")
    parser.add_argument("--disable-final-fusion", action="store_true", help="Ablation: replace with SAM2 box after gate instead of scale-aware fusion.")
    parser.add_argument("--raw-sam-keep-min-trust", type=float, default=0.45)
    parser.add_argument("--raw-sam-keep-min-prompt", type=float, default=0.50)
    parser.add_argument("--raw-sam-keep-min-p2seg-iou", type=float, default=0.20)
    parser.add_argument("--raw-sam-fallback-min-p2seg-iou", type=float, default=0.08)
    parser.add_argument("--raw-sam-small-fallback-center-shift-px", type=float, default=64.0)
    parser.add_argument("--raw-sam-small-light-refine-max-weight", type=float, default=0.06)
    parser.add_argument("--raw-sam-medium-light-refine-max-weight", type=float, default=0.12)
    parser.add_argument("--raw-sam-large-light-refine-max-weight", type=float, default=0.08)
    parser.add_argument("--feature-box-override-thr", type=float, default=0.75)
    parser.add_argument("--disable-feature-final-source", action="store_true", help="Keep feature consistency in trust/gating but never allow feature to become the final box source.")
    parser.add_argument("--feature-sam-disagree-iou", type=float, default=0.40)
    parser.add_argument("--feature-box-fusion-weight", type=float, default=0.50)
    parser.add_argument("--small-fusion-weight-scale", type=float, default=0.50)
    parser.add_argument("--small-safe-max-fusion-weight", type=float, default=0.25)
    parser.add_argument("--small-safe-max-area-ratio", type=float, default=2.0)
    parser.add_argument("--small-final-max-area-ratio", type=float, default=2.5)
    parser.add_argument("--local-density-radius-scale", type=float, default=2.0)
    parser.add_argument("--local-density-norm", type=float, default=4.0)
    parser.add_argument("--route-profile", choices=sorted(ROUTE_PROFILE_OVERRIDES.keys()), default="baseline", help="Routing preset for SAM2 release behavior.")
    parser.add_argument("--uniform-fusion-weight", type=float, default=0.25, help="Frozen P2Seg-to-SAM2 weight used by the uniform_fusion paper baseline.")
    parser.add_argument("--sage-low-sam-score", type=float, default=0.70, help="Frozen SAM2 score below which G1/SAGE-Box preserves the P2Seg box.")
    parser.add_argument("--sage-low-cross-source-iou", type=float, default=0.30, help="Frozen P2Seg-SAM2 mask IoU below which G2/SAGE-Box preserves the P2Seg box.")
    parser.add_argument("--sage-low-prompt-stability", type=float, default=0.50, help="Frozen multi-prompt SAM2 consistency below which G3/SAGE-Box preserves the P2Seg box.")
    parser.add_argument("--ablation", choices=["full", "no_small_safety", "no_scale_cues", "no_reliability_cues", "no_direct_sam", "scpr", "uniform_fusion", "all_direct_sam", "no_scale_conditioning", "no_small_anchor", "no_medium_large_direct", "g1_low_sam_score", "g2_low_cross_source_iou", "g3_low_prompt_stability", "sage_box", "no_prompt_consensus", "no_boundary_calibration"], default="full", help="Paper ablation setting. SAGE variants select one source without using GT.")
    parser.add_argument("--image-ids", default="", help="Comma-separated image ids for debugging.")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--debug-dir", default="")
    parser.add_argument("--debug-count", type=int, default=50)
    parser.add_argument("--debug-candidate-json", default="", help="Optional JSON dump of all scored SAM2 candidates for offline candidate-pool diagnostics.")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress display.")
    return parser.parse_args()


def main():
    args = parse_args()
    apply_route_profile(args)
    refine_results(args)


if __name__ == "__main__":
    main()
