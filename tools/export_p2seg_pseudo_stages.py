"""Export per-stage point-supervised pseudo boxes to COCO-style JSON files."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ann_id_sha256(ann_ids):
    payload = ','.join(str(value) for value in sorted(ann_ids)).encode('ascii')
    return hashlib.sha256(payload).hexdigest()


def expected_annotation_ids(dataset):
    ann_ids = []
    for index in range(len(dataset)):
        values = dataset.get_ann_info(index).get('anns_id', [])
        ann_ids.extend(int(value) for value in values)
    if len(ann_ids) != len(set(ann_ids)):
        raise RuntimeError('The configured export dataset emitted duplicate annotation ids.')
    return ann_ids


def write_contract(dataset, dataset_split, stage, result_path, expected_ids, output_path):
    with Path(result_path).open('r', encoding='utf-8') as handle:
        rows = json.load(handle)
    actual_ids = [row.get('ann_id') for row in rows]
    if None in actual_ids or len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError(f'Stage {stage} result has missing or duplicate annotation ids.')
    if set(actual_ids) != set(expected_ids):
        raise RuntimeError(
            f'Stage {stage} annotation-id coverage mismatch: '
            f'{len(actual_ids)} results != {len(expected_ids)} expected ids.')
    annotation_path = Path(dataset.ann_file).resolve()
    result_path = Path(result_path).resolve()
    marker = {
        'mode': 'p2mnet_export_contract',
        'stage': stage,
        'dataset_split': dataset_split,
        'annotation': str(annotation_path),
        'annotation_sha256': sha256(annotation_path),
        'result': str(result_path),
        'result_sha256': sha256(result_path),
        'image_count': len(dataset),
        'dataset_annotation_count': len(dataset.coco.anns),
        'expected_annotation_count': len(expected_ids),
        'expected_ann_id_sha256': ann_id_sha256(expected_ids),
    }
    output_path = Path(output_path)
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(temporary, output_path)
    return marker


def parse_args():
    parser = argparse.ArgumentParser(description='Export P2Seg pseudo boxes for every stage')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out-prefix', required=True)
    parser.add_argument(
        '--stage', type=int, default=None,
        help='Only retain and write this zero-based pseudo-box stage.')
    parser.add_argument(
        '--dataset-split', choices=['train', 'val', 'test'], required=True,
        help='Config data split used for export. This is required to prevent silent train/val mixups.')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def configure_test_dataset(cfg, dataset_split):
    dataset_cfg = cfg.data[dataset_split]
    if not isinstance(dataset_cfg, dict):
        raise TypeError('Per-stage export requires a single test dataset config.')
    # P2Seg inference needs point boxes, labels, and annotation ids to form
    # proposal bags.  The existing export command requests this explicitly.
    # Point-supervised pseudo-box export needs ann_info at inference time.
    # CocoFmtDataset omits it in ordinary test mode.
    dataset_cfg.pop('keep_test_mode', None)
    dataset_cfg.test_mode = False
    samples_per_gpu = dataset_cfg.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        dataset_cfg.pipeline = replace_ImageToTensor(dataset_cfg.pipeline)
    return dataset_cfg, samples_per_gpu


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.model.pretrained = None
    dataset_cfg, samples_per_gpu = configure_test_dataset(cfg, args.dataset_split)
    dataset = build_dataset(dataset_cfg)
    expected_ids = expected_annotation_ids(dataset)
    print(
        f'dataset_split={args.dataset_split} ann_file={dataset.ann_file} '
        f'img_prefix={dataset.img_prefix} images={len(dataset)} '
        f'expected_annotations={len(expected_ids)}',
        flush=True)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    stage_outputs = None
    with torch.no_grad():
        for data in data_loader:
            batch_stage_results = model(
                return_loss=False,
                rescale=True,
                export_pseudo_stages=True,
                **data)
            if args.stage is not None and not 0 <= args.stage < len(batch_stage_results):
                raise ValueError(
                    f'Requested stage {args.stage}, but model emitted '
                    f'{len(batch_stage_results)} stages.')
            selected = (range(len(batch_stage_results)) if args.stage is None
                        else [args.stage])
            if stage_outputs is None:
                stage_outputs = {stage: [] for stage in selected}
            if set(stage_outputs) != set(selected):
                raise RuntimeError('The number of P2Seg stages changed during export.')
            for stage in selected:
                stage_outputs[stage].extend(batch_stage_results[stage])

    if stage_outputs is None:
        raise RuntimeError('The test dataset is empty.')
    for stage, outputs in stage_outputs.items():
        if len(outputs) != len(dataset):
            raise RuntimeError(f'Stage {stage} emitted {len(outputs)} results for {len(dataset)} images.')
        prefix = f'{args.out_prefix}_stage{stage}'
        result_files, _ = dataset.format_results(outputs, prefix)
        contract_path = f'{prefix}.contract.json'
        write_contract(
            dataset, args.dataset_split, stage, result_files['bbox'],
            expected_ids, contract_path)
        print(f'stage={stage} bbox={result_files["bbox"]} contract={contract_path}')


if __name__ == '__main__':
    main()
