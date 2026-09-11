# CenQ-SAM

Official code release for the CenQ-SAM point-supervised object-detection pipeline. CenQ-SAM refines point-supervised proposals with SAM2 candidates, scores quality and scale consistency, applies conservative small-object safety gates, and exports COCO-format pseudo boxes for detector training. Only the minimal P2Seg inference interface needed to produce candidate masks is retained; no independent segmentation benchmark is included.

## Layout

`mmdet/` and `huicv/` are the MMDetection runtime closure. `configs2/COCO` and `configs2/VOC` contain selected point-supervised, P2Seg inference, and Faster R-CNN configurations. `exp/tools` contains CenQ routing, SAM2 refinement, pseudo-box conversion, router training, and diagnostics. `tools/` contains train/test and export utilities. `scripts/` provides environment-variable driven entry points.

## Environments

The MMDetection stage targets the original OpenMMLab 2.13-era stack (Python 3.7, compatible PyTorch/CUDA, and mmcv-full). Install matching PyTorch/mmcv packages, then run `pip install -v -e .`. Install official SAM2 separately and set `SAM2_ROOT` and `SAM2_CKPT`; checkpoints are not redistributed.

## Data and pipeline

Datasets are not included. COCO should provide `${DATA_ROOT}/train2017`, `${DATA_ROOT}/val2017`, and `${DATA_ROOT}/annotations/`. VOC should provide its normal `VOC2007/` or `VOC2012/` tree and a COCO-format point annotation JSON passed through `ANN`.

`DATA_ROOT=/path/to/coco GPUS=8 bash scripts/train_coco.sh`

`ANN=/path/to/points.json RESULT=/path/to/p2seg_result.json IMG_ROOT=/path/to/coco/train2017 SAM2_ROOT=/path/to/sam2 SAM2_CKPT=/path/to/sam2/checkpoints/model.pt OUT=/path/to/cenq_result.json bash scripts/refine_with_cenq_sam.sh`

`python exp/tools/result2ann_mask2box.py /path/to/points.json /path/to/cenq_result.json /path/to/pseudo_boxes.json --bbox-only --write-refine-diagnostics`

Train the final detector with `tools/train.py`, overriding `data.train.ann_file` to the generated pseudo-box file. Evaluate with `DATA_ROOT=/path/to/coco CHECKPOINT=/path/to/detector.pth bash scripts/eval_coco.sh`; use `scripts/train_voc.sh` and `scripts/eval_voc.sh` for VOC.

## Verification

Run `python -m compileall mmdet huicv exp/tools tools`. All intermediate files are generated outside version control.

## Citation and license

Please cite the CenQ-SAM paper when using this code. MMDetection-derived components retain their Apache 2.0 notices; repository-level scripts follow the included MIT license where applicable.
