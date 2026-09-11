#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANN="${ANN:?Set ANN to the source annotation JSON}"
RESULT="${RESULT:?Set RESULT to the P2Seg/P2MNet result JSON}"
IMG_ROOT="${IMG_ROOT:?Set IMG_ROOT to the image directory}"
SAM2_ROOT="${SAM2_ROOT:?Set SAM2_ROOT to the official SAM2 checkout}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_b+.yaml}"
SAM2_CKPT="${SAM2_CKPT:?Set SAM2_CKPT to a downloaded checkpoint}"
OUT="${OUT:?Set OUT to the refined result JSON}"
cd "${ROOT}"
python exp/tools/refine_p2seg_with_sam2.py --ann "${ANN}" --result "${RESULT}" --img-root "${IMG_ROOT}" --out "${OUT}" --sam2-root "${SAM2_ROOT}" --sam2-config "${SAM2_CONFIG}" --sam2-ckpt "${SAM2_CKPT}" --device "${DEVICE:-cuda}" --candidate-selection-policy "${CANDIDATE_SELECTION_POLICY:-centrality_sam}"
