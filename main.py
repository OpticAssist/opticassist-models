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

def eprint(*values: object, sep: str | None = " ",
           end: str | None = "\n",
           flush: Literal[False] = False):
    print(*values, sep=sep, end=end, flush=flush, file=sys.stderr)


# webcolors doesn't have every single combo of rgb values into text,
# so it just finds the closest one using a loop in its dictionary.
def closest_color_name(rgb):
    min_dist = float('inf')
    closest = "unknown"
    for hex_val, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r, g, b = webcolors.hex_to_rgb(hex_val)
        dist = (r - rgb[0])**2 + (g - rgb[1])**2 + (b - rgb[2])**2
        if dist < min_dist:
            min_dist = dist
            closest = name
    return closest

# uses KMeans cluster. Basically, the image is cropped to 1/4 - 3/4 to center it.
# Then, km basically looks at all the pixels and sorts them to a color group that it fits.
# For example, the image can be 40% Yellow and 60% Green so instead of giving a greenish-yellow,
# it just says green. Makes it more accurate.
def dominant_color(cropped):
    h, w = cropped.shape[:2]
    center = cropped[h//4:3*h//4, w//4:3*w//4]
    if center.size == 0:
        center = cropped

    pixels = center.reshape(-1, 3).astype(float)

    n_clusters = min(3, len(pixels))
    km = KMeans(n_clusters=n_clusters, n_init=3)
    km.fit(pixels)

    counts = np.bincount(km.labels_)
    dominant_bgr = km.cluster_centers_[np.argmax(counts)]
    return (round(dominant_bgr[2]), round(dominant_bgr[1]), round(dominant_bgr[0]))

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


        if cropped_object.size > 0:
            bgr_avg = cv2.mean(cropped_object)[:3]
            rgb_color =  dominant_color(cropped_object)
        else:
            rgb_color = (0,0,0)

        color_name = closest_color_name(rgb_color)

        raw_predictions.append({
            "label": label,
            "confidence": confidence,
            "bounding_box": [x1, y1, x2, y2],
            "color": color_name,
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


