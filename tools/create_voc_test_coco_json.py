#!/usr/bin/env python3
"""Create an annotation-free COCO image manifest for a VOC test split."""

import argparse
import json
from pathlib import Path


VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def parse_args():
    parser = argparse.ArgumentParser(description='Create a COCO image manifest from a VOC test.txt file.')
    parser.add_argument('--image-set', required=True, help='VOC ImageSets/Main/test.txt file.')
    parser.add_argument('--image-dir', required=True, help='Directory containing <image_id>.jpg files.')
    parser.add_argument('--output', required=True, help='Output JSON file.')
    parser.add_argument('--start', type=int, default=0, help='Zero-based start index within test.txt.')
    parser.add_argument('--count', type=int, help='Number of images to include after --start.')
    return parser.parse_args()


def main():
    args = parse_args()
    image_set = Path(args.image_set).resolve()
    image_dir = Path(args.image_dir).resolve()
    output = Path(args.output).resolve()
    image_ids = [line.strip() for line in image_set.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(image_ids) != 10991:
        raise ValueError(f'Expected 10991 VOC2012 test image ids, got {len(image_ids)}')
    missing = [image_id for image_id in image_ids if not (image_dir / f'{image_id}.jpg').is_file()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} test images, first: {missing[0]}.jpg')
    if args.start < 0 or args.start >= len(image_ids):
        raise ValueError(f'--start must be in [0, {len(image_ids) - 1}], got {args.start}')
    selected_ids = image_ids[args.start:] if args.count is None else image_ids[args.start:args.start + args.count]
    if not selected_ids:
        raise ValueError('The requested image slice is empty')
    payload = {
        'images': [
            {'id': index, 'file_name': f'JPEGImages/{image_id}.jpg'}
            for index, image_id in enumerate(selected_ids)
        ],
        'annotations': [],
        'categories': [
            {'id': index + 1, 'name': class_name, 'supercategory': 'none'}
            for index, class_name in enumerate(VOC_CLASSES)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload), encoding='utf-8')
    print(f'Wrote {len(selected_ids)} image entries to {output}')


if __name__ == '__main__':
    main()
