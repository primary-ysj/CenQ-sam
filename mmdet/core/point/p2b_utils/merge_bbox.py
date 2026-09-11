import torch
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
from mmdet.core import multi_apply
import torch.nn.functional as F

_EPS = 1e-6


def _finite_non_negative(scores):
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores.clamp_min(0.0)


def _safe_normalize(scores, dim=1):
    scores = _finite_non_negative(scores)
    denom = scores.sum(dim=dim, keepdim=True)
    if scores.size(dim) == 0:
        return scores
    uniform = scores.new_full(scores.shape, 1.0 / scores.size(dim))
    return torch.where(denom > _EPS, scores / denom.clamp_min(_EPS), uniform)


def _safe_topk(scores, topk):
    if scores.size(1) == 0:
        raise RuntimeError('No proposals available for pruning/topk merge.')
    k = scores.size(1) if topk is None else int(topk)
    k = max(1, min(k, scores.size(1)))
    return _finite_non_negative(scores).topk(k=k, dim=1)


def _gather_candidates(values, indices):
    rows = torch.arange(values.shape[0], device=values.device).unsqueeze(1)
    return values[rows, indices]


def _clamp_boxes(boxes, img_shape):
    if boxes.numel() == 0:
        return boxes
    h, w = img_shape[:2]
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=0.0, neginf=0.0)
    x1 = torch.minimum(boxes[..., 0], boxes[..., 2]).clamp(0, w)
    y1 = torch.minimum(boxes[..., 1], boxes[..., 3]).clamp(0, h)
    x2 = torch.maximum(boxes[..., 0], boxes[..., 2]).clamp(0, w)
    y2 = torch.maximum(boxes[..., 1], boxes[..., 3]).clamp(0, h)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def merge_box_single(cls_score, ins_score, dynamic_weight, gt_point, gt_label, proposals, feat, img_metas, stage_mode,
                     topk, flag=None):
    if stage_mode == 'CBP':
        merge_mode = 'weighted_clsins_topk'
    elif stage_mode == 'PBR':
        if flag == 'iou_pred':
            merge_mode = 'weighted_cls_topk'
        elif flag == 'distill_nms':
            merge_mode = 'weighted_clsins_topk_nms'  ######### changed
        else:
            merge_mode = 'weighted_clsins_topk'

    proposals = proposals.reshape(cls_score.shape[0], cls_score.shape[1], 4)
    proposals = _clamp_boxes(proposals, img_metas['img_shape'])
    num_gt, num_gen = proposals.shape[:2]
    # proposals = proposals.reshape(-1,4)
    if merge_mode == 'weighted_cls_topk':
        cls_score_, idx = _safe_topk(cls_score, topk)
        # weight = cls_score_.unsqueeze(2).repeat([1, 1, 4])
        # weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
        weight = _safe_normalize(cls_score_)
        filtered_boxes = _gather_candidates(proposals, idx)
        boxes = (filtered_boxes * weight[:, :, None]).sum(dim=1)
        boxes = _clamp_boxes(boxes, img_metas['img_shape'])
        # print(weight.sum(dim=1))
        # print(boxes)
        if feat is not None:
            filtered_feat = _gather_candidates(feat, idx)
            feat = (weight[:, :, None] * filtered_feat).sum(1)
        else:
            feat = None
        return boxes, None, None, feat

    if merge_mode == 'weighted_clsins_topk':
        dynamic_weight_, idx = _safe_topk(dynamic_weight, topk)
        # weight = dynamic_weight_.unsqueeze(2).repeat([1, 1, 4])
        # weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
        weight = _safe_normalize(dynamic_weight_)
        filtered_boxes = _gather_candidates(proposals, idx)
        boxes = (filtered_boxes * weight[:, :, None]).sum(dim=1)
        boxes = _clamp_boxes(boxes, img_metas['img_shape'])
        if feat is not None:
            filtered_feat = _gather_candidates(feat, idx)
            feat = (weight[:, :, None] * filtered_feat).sum(1)
        else:
            filtered_feat = None
            feat = None
        # print(weight.sum(dim=1))
        # print(boxes)
        # filtered_scores = dict(cls_score=cls_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
        #                           ins_score=ins_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
        #                        dynamic_weight=dynamic_weight_)
        filtered_scores = dynamic_weight_
        return boxes, filtered_boxes, filtered_scores, filtered_feat

    if merge_mode == 'weighted_clsins_topk_nms':
        dynamic_weight = _finite_non_negative(dynamic_weight)
        wt, id = [], []
        iou = bbox_overlaps(proposals, proposals)
        max_wt, max_id = dynamic_weight.max(dim=1)
        dynamic_weight_ = dynamic_weight
        wt.append(max_wt)
        id.append(max_id)
        row_inds = torch.arange(len(iou), device=proposals.device)
        select_num = max(1, min(int(topk), dynamic_weight.size(1)))
        for i in range(select_num - 1):
            dynamic_weight_ = (iou[row_inds, max_id] < 0.9) * dynamic_weight_
            max_wt, max_id = dynamic_weight_.max(dim=1)
            wt.append(max_wt)
            id.append(max_id)
        wt = torch.stack(wt, dim=1)
        id = torch.stack(id, dim=1)
        # weight = dynamic_weight_.unsqueeze(2).repeat([1, 1, 4])
        # weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
        weight = _safe_normalize(wt)
        filtered_boxes = _gather_candidates(proposals, id)
        boxes = (filtered_boxes * weight[:, :, None]).sum(dim=1)
        boxes = _clamp_boxes(boxes, img_metas['img_shape'])
        if feat is not None:
            filtered_feat = _gather_candidates(feat, id)
            feat = (weight[:, :, None] * filtered_feat).sum(1)
        else:
            filtered_feat = None
            feat = None
        filtered_scores = wt

    return boxes, filtered_boxes, filtered_scores, filtered_feat
    #
    # if merge_mode == 'weighted_clsins_topk_nms':
    #     max_wt, max_id = dynamic_weight.max(dim=1)
    #     best_proposal = proposals[torch.arange(proposals.shape[0]).unsqueeze(1), max_id[:, None]]
    #     iou = bbox_overlaps(proposals, best_proposal)
    #     dynamic_weight_, idx = ((iou < 0.9)[..., 0] * dynamic_weight).topk(k=7, dim=1)
    #     dynamic_weight_ = torch.cat((max_wt[:, None], dynamic_weight_), dim=1)
    #     idx = torch.cat((max_id[:, None], idx), dim=1)
    #     # weight = dynamic_weight_.unsqueeze(2).repeat([1, 1, 4])
    #     # weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
    #     weight = dynamic_weight_ / dynamic_weight_.sum(dim=1, keepdim=True) + 1e-8
    #     filtered_boxes = proposals[torch.arange(proposals.shape[0]).unsqueeze(1), idx]
    #     boxes = (filtered_boxes * weight[:, :, None]).sum(dim=1)
    #     if feat is not None:
    #         filtered_feat = feat[torch.arange(proposals.shape[0]).unsqueeze(1), idx]
    #         feat = (weight[:, :, None] * filtered_feat).sum(1)
    #     else:
    #         feat = None
    #     h, w, _ = img_metas['img_shape']
    #     boxes[:, 0:4:2] = boxes[:, 0:4:2].clamp(0, w)
    #     boxes[:, 1:4:2] = boxes[:, 1:4:2].clamp(0, h)
    #     # print(weight.sum(dim=1))
    #     # print(boxes)
    #     # filtered_scores = dict(cls_score=cls_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
    #     #                        ins_score=ins_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
    #     #                        dynamic_weight=dynamic_weight_)
    #     filtered_scores = dynamic_weight_
    #
    # return boxes, filtered_boxes, filtered_scores, filtered_feat

    if merge_mode == 'weighted_clsins_max_iou':
        dynamic_weight_, idx = _safe_topk(dynamic_weight, topk)
        # weight = dynamic_weight_.unsqueeze(2).repeat([1, 1, 4])
        # weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
        weight = _safe_normalize(dynamic_weight_)
        filtered_boxes = _gather_candidates(proposals, idx)
        boxes = (filtered_boxes * weight[:, :, None]).sum(dim=1)
        boxes = _clamp_boxes(boxes, img_metas['img_shape'])
        if feat is not None:
            filtered_feat = _gather_candidates(feat, idx)
            feat = (weight[:, :, None] * filtered_feat).sum(1)
        else:
            feat = None
        # print(weight.sum(dim=1))
        # print(boxes)
        filtered_scores = dict(cls_score=_gather_candidates(cls_score, idx),
                               ins_score=_gather_candidates(ins_score, idx),
                               dynamic_weight=dynamic_weight_)

        return boxes, filtered_boxes, filtered_scores, feat


def merge_box(bbox_results, proposals_list, proposals_valid_list, gt_labels, gt_bboxes, img_metas, stage_mode, topk,
              proposal_list_base=None, flag=None, metric_bboxes=None):
    """Merge proposal bags into pseudo boxes.

    ``gt_bboxes`` defines the per-image grouping and must be point-derived in
    weak-supervision paths.  ``metric_bboxes`` is an optional diagnostics-only
    target; keeping it separate prevents true boxes from changing proposal
    ordering, weighting, or tensor reshaping.
    """
    cls_scores = bbox_results['cls_score']
    ins_scores = bbox_results['ins_score']
    num_instances = bbox_results['num_instance']
    num_gt = len(gt_labels)

    # num_gt * num_box * num_class
    if stage_mode == 'CBP':
        cls_scores = torch.nan_to_num(cls_scores.softmax(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
    elif stage_mode == 'PBR':
        cls_scores = torch.nan_to_num(cls_scores.sigmoid(), nan=0.0, posinf=0.0, neginf=0.0)
    ins_scores = ins_scores.softmax(dim=-2) * proposals_valid_list
    ins_scores = _safe_normalize(ins_scores, dim=1)
    cls_scores = cls_scores * proposals_valid_list
    dynamic_weight = (cls_scores * ins_scores)
    dynamic_weight = dynamic_weight[torch.arange(len(cls_scores)), :, gt_labels]
    cls_scores = cls_scores[torch.arange(len(cls_scores)), :, gt_labels]
    ins_scores = ins_scores[torch.arange(len(cls_scores)), :, gt_labels]
    # Split by the supervision actually used to construct the proposal bags.
    # A diagnostic target may have the same objects, but must not control this
    # grouping or any model output.
    batch_gt = [len(b) for b in gt_bboxes]
    metric_bboxes = gt_bboxes if metric_bboxes is None else metric_bboxes
    if len(metric_bboxes) != len(gt_bboxes) or sum(batch_gt) != len(gt_labels):
        raise ValueError('Proposal grouping and metric target must have matching image/instance counts.')
    if 'iou_score' in bbox_results:
        if bbox_results['iou_score'] is not None:
            iou_scores = bbox_results['iou_score'].squeeze(-1)
            mean, std = 0.5, 0.5
            iou_scores = iou_scores * std + mean

            iou_min = iou_scores.min(1, keepdim=True)[0]
            iou_max = iou_scores.max(1, keepdim=True)[0]
            iou_scores = (iou_scores - iou_min) / (iou_max - iou_min).clamp_min(_EPS)
            iou_scores = _finite_non_negative(iou_scores)
            flag = 'iou_pred'
            # if bbox_results['obj_score'] is not None:
            #     obj_scores = bbox_results['obj_score']
            #     obj_scores = iou_scores * obj_scores.sigmoid()
            #     if obj_scores.shape[1] != 1:
            #         obj_scores = obj_scores.reshape(*cls_scores.shape, -1)[:, :, 0]
            #
            # else:
            #     obj_scores = obj_scores.reshape(cls_scores.shape).sigmoid()
            # obj_scores = obj_scores.sigmoid()
            cls_scores = cls_scores * iou_scores
            dynamic_weight = dynamic_weight * iou_scores

    if bbox_results['others']:
        feat = bbox_results['others']['x_feat']
        feat = feat.reshape(cls_scores.shape[0], -1, feat.shape[-1])
        feat = torch.split(feat, batch_gt)
    else:
        feat = [None for _ in range(len(batch_gt))]
    # from IPython import embed
    # embed()
    cls_scores = torch.split(cls_scores, batch_gt)
    ins_scores = torch.split(ins_scores, batch_gt)
    gt_labels = torch.split(gt_labels, batch_gt)
    dynamic_weight_list = torch.split(dynamic_weight, batch_gt)
    if not isinstance(proposals_list, list):
        proposals_list = torch.split(proposals_list, batch_gt)
    stage_mode_ = [stage_mode for _ in range(len(cls_scores))]

    topk = [topk for _ in range(len(cls_scores))]
    boxes, filtered_boxes, filtered_scores, feat = multi_apply(merge_box_single, cls_scores, ins_scores,
                                                               dynamic_weight_list,
                                                               gt_bboxes,
                                                               gt_labels,
                                                               proposals_list, feat,
                                                               img_metas, stage_mode_, topk, flag=flag)
    if bbox_results['others']:
        bbox_results['others']['x_feat'] = feat
        bbox_results['others']['filtered_boxes'] = filtered_boxes
        bbox_results['others']['filtered_scores'] = filtered_scores

    pseudo_boxes = torch.cat(boxes).detach()
    # mean_ious =torch.tensor(mean_ious).to(gt_point.device)

    ## condition
    # pseudo_boxes1 = pseudo_boxes * (dynamic_weight.sum(-1,keepdim=True) >0.2)+ torch.cat( proposal_list_base) * (dynamic_weight.sum(-1,keepdim=True)<0.2)

    iou1 = bbox_overlaps(pseudo_boxes, torch.cat(metric_bboxes), is_aligned=True)

    ### scale mean iou
    gt_xywh = bbox_xyxy_to_cxcywh(torch.cat(metric_bboxes))
    scale = gt_xywh[:, 2] * gt_xywh[:, 3]
    mean_iou_s = iou1[scale < 32 ** 2].sum() / (len(iou1[scale < 32 ** 2]) + 1e-5)
    mean_iou_m = iou1[(scale > 32 ** 2) * (scale < 64 ** 2)].sum() / (len(
        iou1[(scale > 32 ** 2) * (scale < 64 ** 2)]) + 1e-5)
    mean_iou_l = iou1[(scale > 64 ** 2) * (scale < 128 ** 2)].sum() / (len(
        iou1[(scale > 64 ** 2) * (scale < 128 ** 2)]) + 1e-5)
    mean_iou_h = iou1[scale > 128 ** 2].sum() / (len(iou1[scale > 128 ** 2]) + 1e-5)

    mean_ious_all = iou1.mean()
    mean_ious = [mean_iou_s, mean_iou_m, mean_iou_l, mean_iou_h, mean_ious_all]
    #
    # if self.test_mean_iou and stage == 1:
    #     self.sum_iou += iou1.sum()
    #     self.sum_num += len(iou1)
    #     # time.sleep(0.01)  # 这里为了查看输出变化，实际使用不需要sleep
    #     print('\r', self.sum_iou / self.sum_num, end='', flush=True)

    pseudo_boxes = torch.split(pseudo_boxes, batch_gt)
    return list(pseudo_boxes), mean_ious, list(filtered_boxes), list(filtered_scores), dynamic_weight.detach()
