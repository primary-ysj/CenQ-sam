#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the VOC root}"
WORK_DIR="${WORK_DIR:-${ROOT}/work_dirs/voc_cenq_sam}"
GPUS="${GPUS:-1}"
ANN="${ANN:?Set ANN to a COCO-format VOC point annotation JSON}"
IMG="${IMG:-${DATA_ROOT}}"
CFG="${ROOT}/configs2/VOC/P2BNet/P2BNet_r50_fpn_1x_VOC_07_ms.py"
cd "${ROOT}"
PORT="${PORT:-29501}" bash tools/dist_train.sh "${CFG}" "${GPUS}" --work-dir "${WORK_DIR}/point_model" --cfg-options "data.train.ann_file=${ANN}" "data.train.img_prefix=${IMG}" "data.val.img_prefix=${IMG}" "data.test.img_prefix=${IMG}"
