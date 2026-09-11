#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the COCO root}"
WORK_DIR="${WORK_DIR:-${ROOT}/work_dirs/coco_cenq_sam}"
GPUS="${GPUS:-1}"
ANN="${ANN:-${DATA_ROOT}/annotations/instances_train2017.json}"
IMG="${IMG:-${DATA_ROOT}/train2017}"
CFG="${ROOT}/configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py"
cd "${ROOT}"
PORT="${PORT:-29500}" bash tools/dist_train.sh "${CFG}" "${GPUS}" --work-dir "${WORK_DIR}/point_model" --cfg-options "data.train.ann_file=${ANN}" "data.train.img_prefix=${IMG}" "data.val.ann_file=${DATA_ROOT}/annotations/instances_val2017.json" "data.val.img_prefix=${DATA_ROOT}/val2017" "data.test.ann_file=${DATA_ROOT}/annotations/instances_val2017.json" "data.test.img_prefix=${DATA_ROOT}/val2017"
