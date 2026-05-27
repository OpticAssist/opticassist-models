import sys
import json
from typing import Literal
import base64
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
import webcolors
from sklearn.cluster import KMeans


# Prints error output to be collected in Rust
def eprint(*values: object, sep: str | None = " ",
           end: str | None = "\n",
           flush: Literal[False] = False):
    print(*values, sep=sep, end=end, flush=flush, file=sys.stderr)


def hsv_to_color_name(hsv_pixel):
    h, s, v = hsv_pixel
    if v < 40:
        return "black"
    if s < 40:
        return "white" if v > 180 else "gray"
    if h < 15 or h >= 165:
        return "red"
    if h < 30:
        return "orange"
    if h < 45:
        return "yellow"
    if h < 90:
        return "green"
    if h < 120:
        return "cyan"
    if h < 135:
        return "blue"
    if h < 150:
        return "purple"
    return "pink"



# uses KMeans cluster. Basically, the image is cropped to 1/4 - 3/4 to center it.
# Then, km basically looks at all the pixels and sorts them to a color group that it fits.
# For example, the image can be 40% Yellow and 60% Green so instead of giving a greenish-yellow,
# it just says green. Makes it more accurate.
def dominant_color(cropped):
    h, w = cropped.shape[:2]
    center = cropped[h//4:3*h//4, w//4:3*w//4]
    if center.size == 0:
        center = cropped
    # hsv is without lighting for better accuracies
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(float)

    n_clusters = min(3, len(pixels))
    km = KMeans(n_clusters=n_clusters, n_init=3)
    km.fit(pixels)

    counts = np.bincount(km.labels_)
    dominant_hsv = km.cluster_centers_[np.argmax(counts)]

    color_name = hsv_to_color_name(dominant_hsv)



    return color_name

def prediction_json(model_output: list[Results], np_img) -> str:
    frame_output = model_output[0]
    raw_predictions: list[dict] = []
    boxes = frame_output.boxes
    for box in boxes:
        label_id = int(box.cls[0])
        label = frame_output.names[label_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        # cropped_object = np_img[int(iy1*1.3):int(iy2//1.3), int(ix1*1.3):int(ix2//1.3)]
        cropped_object = np_img[iy1:iy2, ix1:ix2]
        # cropped_object_color = np_img[iy1*2:iy2//2, ix1*2:ix2//2]
        cropped_object_color = np_img[iy1:iy2, ix1:ix2]




        if cropped_object.size > 0:
            bgr_avg = cv2.mean(cropped_object)[:3]
            color=  dominant_color(cropped_object)
        else:
            color = "unknown"

        raw_predictions.append({
            "label": label,
            "confidence": confidence,
            "bounding_box": [x1, y1, x2, y2],
            "color": color,
        })

    output = {
        "kind": "raw_output",
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
    print(prediction_json(results_arr, np_img), flush=True)

if __name__ == '__main__':
    ready = {
        "kind": "status",
        "message": "200 OK"
    }
    print(ready, flush=True)
    if sys.argv[1]== "run":
        arg = sys.stdin.readline()
        main(arg)
    elif len(sys.argv) > 1:
        eprint("You shouldn't start the model with arguments.")
    while True:
        arg = sys.stdin.readline()
        if arg.strip() != "":
            main(arg)


