import sys
import json
from typing import Literal
import base64
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Prints error output to be collected in Rust
def eprint(*values: object, sep: str | None = " ",
           end: str | None = "\n",
           flush: Literal[False] = False):
    print(*values, sep=sep, end=end, flush=flush, file=sys.stderr)

# Convert a YOLO result to a JSON object
def prediction_json(model_output: list[Results], np_img) -> str:
    frame_output = model_output[0]
    raw_predictions: list[dict] = []
    boxes = frame_output.boxes
    for box in boxes:
        label_id = int(box.cls[0])
        label = frame_output.names[label_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # converts the floats into integers
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

        # crops the image to just get the object using the bounding box
        cropped_object = np_img[iy1:iy2, ix1:ix2]

        # converts the object from bgr to rgb cause cv2 has reverse order, if there's no object it returns no color
        if cropped_object.size > 0:
            bgr_avg = cv2.mean(cropped_object)[:3]
            rgb_color = [round(bgr_avg[2]), round(bgr_avg[1]), round(bgr_avg[0])]
        else:
            rgb_color = [0, 0, 0]
        raw_predictions.append({
            "label": label,
            "confidence": confidence,
            "bounding_box": [x1, y1, x2, y2],
            "raw_rgb": rgb_color,
        })
    output = {
        "kind": "output",
        "image_shape": list(frame_output.orig_shape),
        "raw_predictions": raw_predictions
    }

    return json.dumps(output)


def main(img):
    # convert image into correct np array format
    decoded_bytes = base64.b64decode(img)
    np_arr = np.frombuffer(decoded_bytes, np.uint8)
    np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # start the model
    model = YOLO("yolo26n.pt")

    # get results
    results_arr = model.predict(np_img)

    # send JSON results to stdout
    print(prediction_json(results_arr), flush=True)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        eprint("Expected a base64 image as an argument, got no args.")
