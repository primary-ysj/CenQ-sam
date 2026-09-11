#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the COCO root}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the detector checkpoint}"
cd "${ROOT}"
python tools/test.py "${ROOT}/configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py" "${CHECKPOINT}" --eval bbox --cfg-options "data.test.ann_file=${DATA_ROOT}/annotations/instances_val2017.json" "data.test.img_prefix=${DATA_ROOT}/val2017"
