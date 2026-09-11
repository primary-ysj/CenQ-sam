import json
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

from export_scale_router_mlp_data import route_p2seg_teacher
from refine_p2seg_with_sam2 import (
    ROUTER_FEATURE_KEYS,
    ROUTER_FEATURE_KEYS_V2,
    ROUTER_ROUTE_LABELS,
    ScaleAwareRouter,
    build_scale_priors,
    calibrate_box_with_feature,
    make_refine_info,
    mine_prompt_consensus,
    pass_replace_gate,
    sage_policy,
)


def make_args(router_mlp_json="", use_adaptive_scale_priors=True):
    return Namespace(
        router_mlp_json=router_mlp_json,
        use_adaptive_scale_priors=use_adaptive_scale_priors,
        route_profile="baseline",
        ablation="full",
        scale_head_mode="calibrated",
        scale_small_abs_thr=1024.0,
        scale_large_abs_thr=9216.0,
        scale_small_promote_area=1536.0,
        scale_small_quantile=33.0,
        scale_large_quantile=67.0,
        scale_min_class_count=3,
        small_area_thr=1024.0,
        large_area_thr=9216.0,
        local_density_radius_scale=2.0,
        local_density_norm=4.0,
        small_max_area_ratio=2.0,
        medium_max_area_ratio=4.0,
        large_max_area_ratio=8.0,
        small_center_shift_norm=1.0,
        medium_center_shift_norm=1.5,
        large_center_shift_norm=2.5,
        sam_trust_weight=0.25,
        feature_trust_weight=0.30,
        scale_trust_weight=0.30,
        geometry_trust_weight=0.15,
        scale_prob_trust_weight=0.10,
        density_trust_weight=0.05,
        small_route_penalty_weight=0.20,
        low_trust_thr=0.45,
        high_trust_thr=0.70,
        small_high_trust_thr=0.72,
        small_min_feature_score=0.35,
        small_min_feature_box_iou=0.05,
        small_max_center_shift_px=48.0,
        small_max_fusion_weight=0.35,
        medium_direct_sam_trust_thr=0.80,
        medium_direct_sam_min_prompt=0.60,
        medium_direct_sam_min_p2seg_iou=0.40,
        medium_direct_sam_min_feature_box_iou=0.20,
        medium_max_fusion_weight=0.80,
        large_route_fusion_cap_ceiling=1.0,
        large_direct_sam_trust_thr=0.82,
        large_direct_sam_min_prompt=0.55,
        large_direct_sam_min_p2seg_iou=0.30,
        disable_final_fusion=False,
        final_box_policy="legacy",
        uniform_fusion_weight=0.25,
        sage_low_sam_score=0.70,
        sage_low_cross_source_iou=0.30,
        sage_low_prompt_stability=0.50,
        disable_feature_final_source=False,
        medium_allow_rel_small_abs_medium_direct_sam=False,
        router_mlp_min_route_conf=0.70,
        router_mlp_min_scale_conf=0.60,
        router_mlp_min_release_score=0.60,
        router_mlp_release_trust_delta=0.08,
        router_mlp_release_prompt_delta=0.08,
        router_mlp_release_iou_delta=0.08,
        router_mlp_fusion_delta_scale=0.15,
        router_mlp_max_fusion_cap=0.85,
        gate_min_sam_score=0.75,
        gate_min_area_ratio=0.25,
        gate_min_score_margin=0.05,
        small_gate_min_score_margin=0.05,
        medium_gate_min_score_margin=0.02,
        large_gate_min_score_margin=0.01,
        gate_min_scale_score=0.30,
        gate_min_feature_score=0.20,
        gate_min_p2seg_iou=0.20,
        gate_min_trust_score=0.45,
        gate_max_area_ratio=4.0,
        gate_max_same_class_hits=0,
        gate_max_other_class_hits=999,
        enable_replace_gate=True,
        enable_scale_aware_gate=True,
        feature_box_override_thr=0.75,
        feature_sam_disagree_iou=0.40,
        feature_box_fusion_weight=0.50,
        small_safe_max_fusion_weight=0.25,
        small_safe_max_area_ratio=2.0,
        small_final_max_area_ratio=2.5,
    )


def apply_route_profile(args, profile):
    args.route_profile = profile
    if profile == "v0.3":
        args.medium_gate_min_score_margin = 0.015
        args.large_gate_min_score_margin = 0.008
        args.medium_direct_sam_trust_thr = 0.76
        args.medium_direct_sam_min_prompt = 0.55
        args.medium_direct_sam_min_p2seg_iou = 0.30
        args.medium_direct_sam_min_feature_box_iou = 0.12
        args.large_direct_sam_trust_thr = 0.78
        args.large_direct_sam_min_prompt = 0.50
        args.large_direct_sam_min_p2seg_iou = 0.24
        args.large_route_fusion_cap_ceiling = 0.72
    return args


def make_scale_priors():
    images = {1: {"height": 100, "width": 100}}
    results = [
        {"image_id": 1, "ann_id": 1, "category_id": 1, "_p2seg_area": 64.0},
        {"image_id": 1, "ann_id": 2, "category_id": 1, "_p2seg_area": 400.0},
        {"image_id": 1, "ann_id": 3, "category_id": 1, "_p2seg_area": 2500.0},
        {"image_id": 1, "ann_id": 4, "category_id": 1, "_p2seg_area": 6400.0},
    ]
    return {"scale": build_scale_priors(results, images, 33.0, 67.0)}


def make_large_object_scale_priors():
    images = {1: {"height": 100, "width": 100}}
    results = [
        {"image_id": 1, "ann_id": 1, "category_id": 1, "_p2seg_area": 2500.0},
        {"image_id": 1, "ann_id": 2, "category_id": 1, "_p2seg_area": 5000.0},
        {"image_id": 1, "ann_id": 3, "category_id": 1, "_p2seg_area": 8000.0},
        {"image_id": 1, "ann_id": 4, "category_id": 1, "_p2seg_area": 9000.0},
    ]
    return {"scale": build_scale_priors(results, images, 33.0, 67.0)}


def make_candidate_and_context():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 20:60] = True
    p2seg_box = np.asarray([20.0, 20.0, 60.0, 60.0], dtype=np.float32)
    candidate = {"mask": mask, "sam_score": 0.92, "prompt": "synthetic_point_box"}
    score = {
        "sam_score": 0.92,
        "total": 3.5,
        "total_score": 3.5,
        "score_margin": 0.5,
        "area_ratio": 1.0,
        "p2seg_iou": 1.0,
        "feature_available": True,
        "feature_consistency_score": 0.90,
        "feature_support_ratio": 0.75,
        "feature_box_iou": 0.90,
        "feature_box": [21.0, 21.0, 59.0, 59.0],
        "prompt_consistency_score": 0.95,
        "same_class_hits": 0,
        "other_class_hits": 0,
        "same_class_point_hits": 0,
        "other_class_point_hits": 0,
        "quality": 0.95,
    }
    context = {
        "img_info": {"id": 1, "height": 100, "width": 100},
        "ann": {"id": 1001, "image_id": 1, "category_id": 1, "bbox": [20.0, 20.0, 40.0, 40.0], "point": [40.0, 40.0]},
        "pred": {"image_id": 1, "ann_id": 1001, "category_id": 1, "bbox": [20.0, 20.0, 40.0, 40.0]},
        "point": [40.0, 40.0],
        "other_anns": [],
        "image_preds": [],
    }
    return candidate, score, p2seg_box, context


def make_feature_route_mlp(path):
    hidden = 1
    model = {
        "feature_keys": ROUTER_FEATURE_KEYS,
        "mean": [0.0] * len(ROUTER_FEATURE_KEYS),
        "std": [1.0] * len(ROUTER_FEATURE_KEYS),
        "hidden_weight": [[0.0] * len(ROUTER_FEATURE_KEYS) for _ in range(hidden)],
        "hidden_bias": [0.0] * hidden,
        "scale_weight": [[0.0] * hidden for _ in range(3)],
        "scale_bias": [-20.0, 20.0, -20.0],
        "route_weight": [[0.0] * hidden for _ in range(4)],
        "route_bias": [-20.0, -20.0, -20.0, 20.0],
        "fusion_weight": [0.0] * hidden,
        "fusion_bias": 0.0,
    }
    Path(path).write_text(json.dumps(model), encoding="utf-8")


def assert_required_refine_fields(refine_info):
    required = {
        "scale_group",
        "scale_prob",
        "sam2_trust_score",
        "feature_consistency_score",
        "scale_consistency_score",
        "fusion_weight",
        "final_bbox",
        "final_source",
        "gate_reason",
        "scale_abs_group",
        "scale_relative_group",
        "scale_ambiguity",
        "scale_head_reason",
        "route_profile",
        "route_scale_level",
        "route_support_score",
        "route_effective_trust",
        "route_dynamic_fusion_cap",
        "scale_safety_group",
        "p2seg_log_area_norm",
        "candidate_log_area_norm",
        "safe_sam_log_area_norm",
        "safe_feature_log_area_norm",
        "p2seg_aspect_ratio",
    }
    missing = sorted(required.difference(refine_info.keys()))
    assert not missing, f"missing refine fields: {missing}"
    assert refine_info["scale_group"] in {"small", "medium", "large"}
    assert refine_info["final_source"] in ROUTER_ROUTE_LABELS
    assert refine_info["final_bbox"] is not None and len(refine_info["final_bbox"]) == 4
    assert refine_info["scale_abs_group"] in {"small", "medium", "large", "unknown"}
    assert refine_info["scale_relative_group"] in {"small", "medium", "large", "unknown"}


def make_v2_release_mlp(path):
    hidden = 1
    model = {
        "router_head_version": 2,
        "feature_keys": ROUTER_FEATURE_KEYS_V2,
        "mean": [0.0] * len(ROUTER_FEATURE_KEYS_V2),
        "std": [1.0] * len(ROUTER_FEATURE_KEYS_V2),
        "hidden_weight": [[0.0] * len(ROUTER_FEATURE_KEYS_V2) for _ in range(hidden)],
        "hidden_bias": [1.0] * hidden,
        "scale_weight": [[0.0] * hidden for _ in range(3)],
        "scale_bias": [-20.0, -20.0, 20.0],
        "release_weight": [10.0] * hidden,
        "release_bias": 10.0,
        "fusion_delta_weight": [10.0] * hidden,
        "fusion_delta_bias": 10.0,
    }
    Path(path).write_text(json.dumps(model), encoding="utf-8")


def smoke_adaptive_scale_priors():
    router = ScaleAwareRouter(make_args(use_adaptive_scale_priors=True), make_scale_priors())
    small_group, _, small_meta = router.group_from_area(64.0, 1, 10000.0)
    large_group, _, large_meta = router.group_from_area(6400.0, 1, 10000.0)
    assert small_group == "small", (small_group, small_meta)
    assert large_group == "medium", (large_group, large_meta)
    assert small_meta["scale_mode"] == "calibrated"
    assert large_meta["scale_abs_group"] == "medium"


def smoke_abs_medium_blocks_relative_small():
    router = ScaleAwareRouter(make_args(use_adaptive_scale_priors=True), make_large_object_scale_priors())
    group, _, meta = router.group_from_area(2500.0, 1, 10000.0)
    assert group == "medium", (group, meta)
    assert meta["scale_abs_group"] == "medium", meta
    assert meta["scale_relative_group"] == "small", meta
    assert meta["scale_ambiguity"] == "rel_small_abs_medium", meta


def smoke_v0_refine_payload():
    router = ScaleAwareRouter(make_args(use_adaptive_scale_priors=False), {})
    candidate, score, p2seg_box, context = make_candidate_and_context()
    route_info = router.decode_final_box(candidate, score, p2seg_box, context)
    score.update(route_info)
    refine_info = make_refine_info(True, "pass", candidate, score, None, "synthetic")
    assert_required_refine_fields(refine_info)


def smoke_medium_margin_gate_relaxed():
    args = make_args(use_adaptive_scale_priors=False)
    candidate, score, _, _ = make_candidate_and_context()
    score.update({
        "final_source": "fusion",
        "scale_group": "medium",
        "sam2_trust_score": 0.92,
        "scale_consistency_score": 0.90,
    })
    second_score = {"total": score["total"] - 0.03}
    gate_ok, gate_reason = pass_replace_gate(candidate, score, second_score, args)
    assert gate_ok, (gate_reason, score)
    assert gate_reason == "pass", gate_reason
    assert score["gate_score_margin_thr"] == args.medium_gate_min_score_margin, score


def smoke_medium_rel_large_can_direct_sam():
    router = ScaleAwareRouter(make_args(use_adaptive_scale_priors=False), {})
    sam_box = np.asarray([18.0, 18.0, 62.0, 62.0], dtype=np.float32)
    p2seg_box = np.asarray([20.0, 20.0, 60.0, 60.0], dtype=np.float32)
    score = {
        "scale_ambiguity": "rel_large_abs_medium",
        "prompt_consistency_score": 0.72,
        "p2seg_iou": 0.66,
        "feature_available": False,
    }
    final_box, source, fusion_weight = router._decode_medium_box(0.86, sam_box, p2seg_box, score)
    assert source == "sam2", (source, fusion_weight, final_box)
    assert fusion_weight == 1.0, fusion_weight


def smoke_medium_dynamic_route_stats():
    router = ScaleAwareRouter(make_args(use_adaptive_scale_priors=False), {})
    score = {
        "scale_candidate_area": 8000.0,
        "scale_prob": 0.86,
        "prompt_consistency_score": 0.82,
        "p2seg_iou": 0.74,
        "feature_available": True,
        "feature_box_iou": 0.68,
        "scale_ambiguity": "rel_large_abs_medium",
        "local_point_density": 0.05,
    }
    stats = router._build_route_stats("medium", 0.76, score)
    assert stats["route_profile"] == "medium_dynamic", stats
    assert stats["route_effective_trust"] > 0.76, stats
    assert stats["route_direct_trust_thr"] < router.args.medium_direct_sam_trust_thr, stats
    assert stats["route_dynamic_fusion_cap"] > 0.6, stats


def smoke_v03_releases_medium_and_large_more_than_baseline():
    baseline_router = ScaleAwareRouter(make_args(use_adaptive_scale_priors=False), {})
    v03_router = ScaleAwareRouter(apply_route_profile(make_args(use_adaptive_scale_priors=False), "v0.3"), {})
    medium_score = {
        "scale_ambiguity": "rel_large_abs_medium",
        "prompt_consistency_score": 0.58,
        "p2seg_iou": 0.36,
        "feature_available": True,
        "feature_box_iou": 0.16,
    }
    p2seg_box = np.asarray([20.0, 20.0, 60.0, 60.0], dtype=np.float32)
    sam_box = np.asarray([18.0, 18.0, 62.0, 62.0], dtype=np.float32)
    baseline_box, baseline_source, baseline_weight = baseline_router._decode_medium_box(0.77, sam_box, p2seg_box, medium_score)
    v03_box, v03_source, v03_weight = v03_router._decode_medium_box(0.77, sam_box, p2seg_box, medium_score)
    assert baseline_source == "fusion", (baseline_source, baseline_weight, baseline_box)
    assert v03_source == "sam2", (v03_source, v03_weight, v03_box)
    large_stats = v03_router._build_route_stats("large", 0.80, {
        "scale_candidate_area": 12000.0,
        "scale_prob": 0.92,
        "prompt_consistency_score": 0.86,
        "p2seg_iou": 0.78,
        "feature_available": False,
        "scale_ambiguity": "rel_large_abs_large",
        "local_point_density": 0.02,
    })
    assert large_stats["route_dynamic_fusion_cap"] <= 0.72, large_stats


def smoke_small_route_stays_conservative():
    router = ScaleAwareRouter(make_args(use_adaptive_scale_priors=False), {})
    score = {
        "scale_candidate_area": 800.0,
        "scale_prob": 0.10,
        "prompt_consistency_score": 0.80,
        "p2seg_iou": 0.78,
        "feature_available": True,
        "feature_box_iou": 0.70,
        "scale_ambiguity": "boundary",
        "local_point_density": 0.20,
    }
    stats = router._build_route_stats("small", 0.74, score)
    assert stats["route_profile"] == "small_conservative", stats
    assert stats["route_effective_trust"] <= 0.74, stats
    assert stats["route_dynamic_fusion_cap"] <= router.args.small_max_fusion_weight, stats


def smoke_mlp_missing_feature_box_reject():
    candidate, score, p2seg_box, context = make_candidate_and_context()
    score.pop("feature_box", None)
    with tempfile.TemporaryDirectory() as tmpdir:
        mlp_path = Path(tmpdir) / "feature_route_mlp.json"
        make_feature_route_mlp(mlp_path)
        router = ScaleAwareRouter(make_args(router_mlp_json=str(mlp_path), use_adaptive_scale_priors=False), {})
        route_info = router.decode_final_box(candidate, score, p2seg_box, context)
    assert route_info.get("router_mlp_used") is False, route_info
    assert route_info.get("router_mlp_reject_reason") == "missing_feature_box", route_info
    assert route_info.get("final_source") in ROUTER_ROUTE_LABELS, route_info


def smoke_small_safety_blocks_direct_sam():
    mask = np.zeros((100, 100), dtype=bool)
    mask[22:54, 22:54] = True
    p2seg_box = np.asarray([30.0, 30.0, 50.0, 50.0], dtype=np.float32)
    candidate = {"mask": mask, "sam_score": 0.98, "prompt": "small_safe"}
    score = {
        "sam_score": 0.98,
        "total": 4.0,
        "total_score": 4.0,
        "score_margin": 0.5,
        "area_ratio": 2.56,
        "p2seg_iou": 0.39,
        "feature_available": False,
        "feature_consistency_score": 0.0,
        "feature_support_ratio": 0.0,
        "feature_box_iou": 0.0,
        "prompt_consistency_score": 0.95,
        "same_class_hits": 0,
        "other_class_hits": 0,
        "same_class_point_hits": 0,
        "other_class_point_hits": 0,
        "quality": 0.95,
    }
    context = {
        "img_info": {"id": 1, "height": 100, "width": 100},
        "ann": {"id": 1002, "image_id": 1, "category_id": 1, "bbox": [30.0, 30.0, 20.0, 20.0], "point": [40.0, 40.0]},
        "pred": {"image_id": 1, "ann_id": 1002, "category_id": 1, "bbox": [30.0, 30.0, 20.0, 20.0]},
        "point": [40.0, 40.0],
        "other_anns": [],
        "image_preds": [],
    }
    router = ScaleAwareRouter(make_args(use_adaptive_scale_priors=False), {})
    route_info = router.decode_final_box(candidate, score, p2seg_box, context)
    assert route_info["scale_safety_group"] == "small", route_info
    assert route_info["final_source"] != "sam2", route_info
    assert route_info["fusion_weight"] <= router.args.small_safe_max_fusion_weight + 1e-6, route_info


def smoke_paper_ablation_switches_change_routing():
    sam_box = np.asarray([18.0, 18.0, 62.0, 62.0], dtype=np.float32)
    p2seg_box = np.asarray([20.0, 20.0, 60.0, 60.0], dtype=np.float32)
    medium_score = {
        "scale_ambiguity": "none",
        "prompt_consistency_score": 0.0,
        "p2seg_iou": 0.0,
        "feature_available": False,
    }
    no_direct_args = make_args(use_adaptive_scale_priors=False)
    no_direct_args.ablation = "no_direct_sam"
    no_direct_router = ScaleAwareRouter(no_direct_args, {})
    _, source, _ = no_direct_router._decode_medium_box(0.95, sam_box, p2seg_box, medium_score)
    assert source == "fusion", source

    no_reliability_args = make_args(use_adaptive_scale_priors=False)
    no_reliability_args.ablation = "no_reliability_cues"
    no_reliability_router = ScaleAwareRouter(no_reliability_args, {})
    _, source, _ = no_reliability_router._decode_medium_box(0.95, sam_box, p2seg_box, medium_score)
    assert source == "sam2", source

    no_scale_args = make_args(use_adaptive_scale_priors=False)
    no_scale_args.ablation = "no_scale_cues"
    no_scale_router = ScaleAwareRouter(no_scale_args, {})
    stats = no_scale_router._build_route_stats("small", 0.70, medium_score)
    assert stats["route_profile"] == "uniform", stats

    uniform_args = make_args(use_adaptive_scale_priors=False)
    uniform_args.ablation = "uniform_fusion"
    uniform_router = ScaleAwareRouter(uniform_args, {})
    uniform_context = {
        "img_info": {"id": 1, "height": 100, "width": 100},
        "ann": {"id": 1004, "image_id": 1, "category_id": 1, "bbox": [20.0, 20.0, 40.0, 40.0], "point": [40.0, 40.0]},
        "pred": {"image_id": 1, "ann_id": 1004, "category_id": 1, "bbox": [20.0, 20.0, 40.0, 40.0]},
        "point": [40.0, 40.0], "other_anns": [], "image_preds": [],
    }
    uniform_candidate = {"mask": np.pad(np.ones((40, 40), dtype=bool), ((10, 50), (10, 50))), "sam_score": 0.9, "prompt": "uniform"}
    uniform_score = {"feature_available": False, "p2seg_iou": 0.5, "area_ratio": 1.0, "prompt_consistency_score": 1.0, "same_class_hits": 0, "other_class_hits": 0}
    uniform_route = uniform_router.decode_final_box(uniform_candidate, uniform_score, p2seg_box, uniform_context)
    assert uniform_route["final_source"] == "fusion", uniform_route
    assert abs(uniform_route["fusion_weight"] - 0.25) < 1e-6, uniform_route

    small_mask = np.zeros((100, 100), dtype=bool)
    small_mask[22:54, 22:54] = True
    small_candidate = {"mask": small_mask, "sam_score": 0.98, "prompt": "small_ablation"}
    small_p2seg_box = np.asarray([30.0, 30.0, 50.0, 50.0], dtype=np.float32)
    small_score = {
        "sam_score": 0.98, "total": 4.0, "total_score": 4.0, "score_margin": 0.5,
        "area_ratio": 2.56, "p2seg_iou": 0.39, "feature_available": False,
        "feature_consistency_score": 0.0, "feature_support_ratio": 0.0, "feature_box_iou": 0.0,
        "prompt_consistency_score": 0.95, "same_class_hits": 0, "other_class_hits": 0,
        "same_class_point_hits": 0, "other_class_point_hits": 0, "quality": 0.95,
    }
    small_context = {
        "img_info": {"id": 1, "height": 100, "width": 100},
        "ann": {"id": 1003, "image_id": 1, "category_id": 1, "bbox": [30.0, 30.0, 20.0, 20.0], "point": [40.0, 40.0]},
        "pred": {"image_id": 1, "ann_id": 1003, "category_id": 1, "bbox": [30.0, 30.0, 20.0, 20.0]},
        "point": [40.0, 40.0], "other_anns": [], "image_preds": [],
    }
    no_small_args = make_args(use_adaptive_scale_priors=False)
    no_small_args.ablation = "no_small_safety"
    no_small_router = ScaleAwareRouter(no_small_args, {})
    route_info = no_small_router.decode_final_box(small_candidate, small_score, small_p2seg_box, small_context)
    assert route_info["scale_safety_group"] == "medium", route_info


def smoke_sage_policy_variants_are_isolated():
    sam_box = np.asarray([18.0, 18.0, 62.0, 62.0], dtype=np.float32)
    p2seg_box = np.asarray([20.0, 20.0, 60.0, 60.0], dtype=np.float32)
    safe_score = {"sam_score": 0.92, "p2seg_iou": 0.85, "prompt_consistency_score": 0.90}
    cases = [
        ("g1_low_sam_score", {**safe_score, "sam_score": 0.60}, "p2seg", "low_sam_score"),
        ("g2_low_cross_source_iou", {**safe_score, "p2seg_iou": 0.20}, "p2seg", "low_cross_source_iou"),
        ("g3_low_prompt_stability", {**safe_score, "prompt_consistency_score": 0.40}, "p2seg", "low_prompt_stability"),
        ("sage_box", {**safe_score, "sam_score": 0.60}, "p2seg", "low_sam_score"),
        ("sage_box", safe_score, "sam2", "sam2_selected"),
    ]
    for mode, score, expected_source, expected_reason in cases:
        args = make_args(use_adaptive_scale_priors=False)
        args.ablation = mode
        source, reasons = sage_policy(args, score, sam_box, p2seg_box)
        assert source == expected_source, (mode, source, reasons)
        assert expected_reason in reasons, (mode, reasons)

    args = make_args(use_adaptive_scale_priors=False)
    args.ablation = "g1_low_sam_score"
    keep_p2seg, gate_reason = pass_replace_gate(
        {"sam_score": 0.60}, {"final_source": "p2seg"}, None, args)
    assert keep_p2seg and gate_reason == "ablation_g1_low_sam_score_p2seg", gate_reason


def smoke_v2_release_only_medium_large():
    candidate, score, p2seg_box, context = make_candidate_and_context()
    with tempfile.TemporaryDirectory() as tmpdir:
        mlp_path = Path(tmpdir) / "release_mlp.json"
        make_v2_release_mlp(mlp_path)
        router = ScaleAwareRouter(make_args(router_mlp_json=str(mlp_path), use_adaptive_scale_priors=False), {})
        route_info = router.decode_final_box(candidate, score, p2seg_box, context)
    assert route_info.get("router_head_version") == 2, route_info
    assert route_info.get("router_mlp_used") is True, route_info
    assert route_info.get("perceived_scale_group") == "large", route_info


def smoke_route_p2seg_teacher():
    assert route_p2seg_teacher({"selected": False, "gate_reason": "route_p2seg", "final_source": "p2seg"})
    assert not route_p2seg_teacher({"selected": True, "gate_reason": "pass", "final_source": "sam2"})


def smoke_prompt_consensus_mines_point_connected_mask():
    args = Namespace(
        enable_prompt_consensus=True,
        consensus_topk=3,
        consensus_min_iou=0.45,
        consensus_min_containment=0.80,
        consensus_min_support=0.55,
        consensus_temperature=0.20,
        consensus_max_expansion=1.50,
    )
    first = np.zeros((100, 100), dtype=bool)
    second = np.zeros((100, 100), dtype=bool)
    first[20:60, 20:60] = True
    second[22:62, 22:62] = True
    consensus = mine_prompt_consensus(
        [
            ({"mask": first, "sam_score": 0.95, "prompt": "point"}, {"total": 4.0}),
            ({"mask": second, "sam_score": 0.92, "prompt": "point_box"}, {"total": 3.9}),
        ],
        [40.0, 40.0],
        args,
    )
    assert consensus is not None
    assert consensus["prompt"] == "prompt_consensus"
    assert consensus["consensus_member_count"] == 2
    assert consensus["mask"][40, 40]


def smoke_boundary_calibration_keeps_small_untouched():
    args = Namespace(
        enable_boundary_calibration=True,
        boundary_calibration_min_feature_score=0.70,
        boundary_calibration_min_box_iou=0.30,
        boundary_calibration_max_area_ratio=2.0,
        boundary_calibration_min_weight=0.08,
        boundary_calibration_max_weight=0.35,
    )
    sam_box = np.asarray([20.0, 20.0, 60.0, 60.0], dtype=np.float32)
    feature_box = np.asarray([24.0, 24.0, 64.0, 64.0], dtype=np.float32)
    score = {"feature_available": True, "feature_consistency_score": 0.88, "feature_box_iou": 0.68}
    calibrated, info = calibrate_box_with_feature(sam_box, feature_box, score, "medium", {"width": 100, "height": 100}, args)
    assert info["boundary_calibration_applied"]
    assert calibrated[0] > sam_box[0] and calibrated[2] > sam_box[2]
    small_box, small_info = calibrate_box_with_feature(sam_box, feature_box, score, "small", {"width": 100, "height": 100}, args)
    assert not small_info["boundary_calibration_applied"]
    assert np.array_equal(small_box, sam_box)


def main():
    smoke_adaptive_scale_priors()
    smoke_abs_medium_blocks_relative_small()
    smoke_v0_refine_payload()
    smoke_medium_margin_gate_relaxed()
    smoke_medium_rel_large_can_direct_sam()
    smoke_medium_dynamic_route_stats()
    smoke_v03_releases_medium_and_large_more_than_baseline()
    smoke_small_route_stays_conservative()
    smoke_mlp_missing_feature_box_reject()
    smoke_small_safety_blocks_direct_sam()
    smoke_paper_ablation_switches_change_routing()
    smoke_sage_policy_variants_are_isolated()
    smoke_v2_release_only_medium_large()
    smoke_route_p2seg_teacher()
    smoke_prompt_consensus_mines_point_connected_mask()
    smoke_boundary_calibration_keeps_small_untouched()
    print("smoke_scale_router_minimal_ok")


if __name__ == "__main__":
    main()
