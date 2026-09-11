#!/usr/bin/env python3
"""Convert MMDetection detection outputs into a Pascal VOC submission."""

import argparse
import pickle
import tarfile
from pathlib import Path


VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection outputs to VOC detection submission files.')
    parser.add_argument('--result', required=True, help='Pickle file from tools/test.py --out.')
    parser.add_argument('--image-set', required=True, help='VOC ImageSets/Main/test.txt file.')
    parser.add_argument('--output-dir', required=True, help='Directory that will contain results/VOC2012/Main.')
    parser.add_argument('--competition', type=int, default=3, choices=(3, 4), help='VOC detection competition number.')
    parser.add_argument('--archive', help='Optional .tar.gz archive path for upload.')
    return parser.parse_args()


def read_image_ids(path):
    image_ids = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not image_ids:
        raise ValueError(f'No image ids found in {path}')
    return image_ids


def load_results(path):
    with path.open('rb') as handle:
        results = pickle.load(handle)
    if not isinstance(results, list):
        raise TypeError(f'Expected a list of per-image results, got {type(results).__name__}')
    return results


def bbox_results(result):
    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, (list, tuple)):
        raise TypeError(f'Expected class-wise bbox results, got {type(result).__name__}')
    if len(result) != len(VOC_CLASSES):
        raise ValueError(f'Expected {len(VOC_CLASSES)} classes, got {len(result)}')
    return result


def write_submission(results, image_ids, output_dir, competition):
    main_dir = output_dir / 'results' / 'VOC2012' / 'Main'
    main_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        class_name: (main_dir / f'comp{competition}_det_test_{class_name}.txt').open('w', encoding='utf-8')
        for class_name in VOC_CLASSES
    }
    detection_count = 0
    try:
        for image_id, result in zip(image_ids, results):
            for class_name, boxes in zip(VOC_CLASSES, bbox_results(result)):
                for box in boxes:
                    if len(box) < 5:
                        raise ValueError(f'Malformed detection for image {image_id}: {box}')
                    # MMDetection boxes are zero-based; the VOC devkit expects one-based coordinates.
                    x1, y1, x2, y2, score = (float(value) for value in box[:5])
                    handles[class_name].write(f'{image_id} {score:.8f} {x1 + 1.0:.4f} {y1 + 1.0:.4f} {x2 + 1.0:.4f} {y2 + 1.0:.4f}\n')
                    detection_count += 1
    finally:
        for handle in handles.values():
            handle.close()
    return main_dir, detection_count


def create_archive(output_dir, archive_path):
    results_dir = output_dir / 'results'
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, 'w:gz') as archive:
        archive.add(results_dir, arcname='results')


def main():
    args = parse_args()
    result_path = Path(args.result).resolve()
    image_set_path = Path(args.image_set).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    if not image_set_path.is_file():
        raise FileNotFoundError(image_set_path)
    image_ids = read_image_ids(image_set_path)
    results = load_results(result_path)
    if len(results) != len(image_ids):
        raise ValueError(f'Result/image count mismatch: {len(results)} results for {len(image_ids)} test images')
    main_dir, detection_count = write_submission(results, image_ids, output_dir, args.competition)
    print(f'Wrote {detection_count} detections to {main_dir}')
    if args.archive:
        archive_path = Path(args.archive).resolve()
        create_archive(output_dir, archive_path)
        print(f'Created upload archive: {archive_path}')


if __name__ == '__main__':
    main()
