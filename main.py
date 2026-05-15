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
def prediction_json(model_output: list[Results]) -> str:
    frame_output = model_output[0]
    detections: list[dict] = []
    boxes = frame_output.boxes
    for box in boxes:
        label_id = int(box.cls[0])
        label = frame_output.names[label_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "label": label,
            "confidence": confidence,
            "bounding_box": [x1, y1, x2, y2],
        })
    output = {
        "kind": "output",
        "image_shape": list(frame_output.orig_shape),
        "detections": detections
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
