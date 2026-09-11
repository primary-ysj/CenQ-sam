import argparse
import json

from pycocotools import mask as mask_utils


def encode_rectangle(bbox, height, width):
    x, y, box_width, box_height = [float(value) for value in bbox]
    left = max(0, min(width, int(x)))
    top = max(0, min(height, int(y)))
    right = max(left + 1, min(width, int(x + box_width)))
    bottom = max(top + 1, min(height, int(y + box_height)))
    polygon = [[left, top, right, top, right, bottom, left, bottom]]
    rle = mask_utils.merge(mask_utils.frPyObjects(polygon, height, width))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation")
    parser.add_argument("bbox_result")
    parser.add_argument("output")
    args = parser.parse_args()

    with open(args.annotation, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    with open(args.bbox_result, "r", encoding="utf-8") as handle:
        predictions = json.load(handle)
    images = {image["id"]: image for image in dataset["images"]}
    output = []
    missing_ann_id = 0
    for prediction in predictions:
        image = images[prediction["image_id"]]
        converted = dict(prediction)
        converted["segmentation"] = encode_rectangle(prediction["bbox"], image["height"], image["width"])
        if "ann_id" not in converted:
            missing_ann_id += 1
        output.append(converted)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle)
    print(f"predictions={len(output)}, missing_ann_id={missing_ann_id}, output={args.output}")


if __name__ == "__main__":
    main()
