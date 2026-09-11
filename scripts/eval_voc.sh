#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANN="${ANN:?Set ANN to the VOC evaluation annotation JSON}"
IMG="${IMG:?Set IMG to the VOC image root}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the detector checkpoint}"
cd "${ROOT}"
python tools/test.py "${ROOT}/configs2/VOC/detection/faster_rcnn_r50_fpn_3x_VOC_ms.py" "${CHECKPOINT}" --eval bbox --cfg-options "data.test.ann_file=${ANN}" "data.test.img_prefix=${IMG}"
