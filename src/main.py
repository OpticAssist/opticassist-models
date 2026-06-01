import sys
import json
import base64
from json import JSONDecodeError
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
from sklearn.cluster import KMeans
from abc import ABC
from dataclasses import dataclass, asdict
import logging

logging.disable(logging.CRITICAL)
sys.stdout.reconfigure(line_buffering=True) # Removes the need for `flush = True` with every print

@dataclass(slots=True, kw_only=True)
class Message(ABC):
    kind: str

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict())

    def __str__(self) -> str:
        return self.to_json()

@dataclass(slots=True, kw_only=True)
class Status(Message):
    message: str
    kind: str = "status"

@dataclass(slots=True, kw_only=True)
class Input(Message):
    image: str
    kind: str = "input"

@dataclass(slots=True, kw_only=True)
class Error(Message):
    message: str
    kind: str = "error"

@dataclass(slots=True, kw_only=True)
class RawPrediction:
    label: str
    confidence: float
    bounding_box: list[int]
    color: str


@dataclass(slots=True, kw_only=True)
class RawOutput(Message):
    image_shape: list[int]
    raw_predictions: list[RawPrediction]
    kind: str = "raw_output"

def hsv_to_color_name(hsv_pixel):
    h, s, v = hsv_pixel
    if v < 40:
        return "black"
    if s < 40:
        return "white" if v > 180 else "gray"
    if h < 15 or h >= 165:
        return "red"
    if h < 25:
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

# Uses KMeans cluster. Basically, the image is cropped to 1/4 -- 3/4 to center it.
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
    raw_predictions: list[RawPrediction] = []
    boxes = frame_output.boxes
    for box in boxes:
        label_id = int(box.cls[0])
        label = frame_output.names[label_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

        bounding_box = [ix1, iy1, ix2, iy2]
        cropped_object = np_img[iy1:iy2, ix1:ix2]

        if cropped_object.size > 0:
            # bgr_avg = cv2.mean(cropped_object)[:3]
            color=  dominant_color(cropped_object)
        else:
            color = "unknown"

        raw_predictions.append(RawPrediction(label=label, confidence=confidence, bounding_box=bounding_box, color=color))

    image_shape = list(frame_output.orig_shape)
    output = RawOutput(image_shape=image_shape, raw_predictions=raw_predictions)

    return str(output)

model = YOLO("yolo26n.onnx", task="detect", verbose=False)
print(Status(message="200 OK"))

def main(img: str):

    # convert image into correct np array format
    decoded_bytes = base64.b64decode(img)

    np_arr = np.frombuffer(decoded_bytes, np.uint8)
    np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # get results
    results_arr = model.predict(np_img, verbose=False)

    # send JSON results to stdout
    print(prediction_json(results_arr, np_img))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        arg = sys.stdin.readline()
        try:
            main(arg)
        except Exception as e:
            print(Error(message=f"Model crashed while running, skipping the frame: {e}"))
        sys.exit(0)
    while True:
        arg = sys.stdin.readline()
        if arg.strip() != "":
            try:
                data = json.loads(arg)
            except JSONDecodeError as e:
                print(Error(message=f"Failed to decode JSON, skipping the frame: {e}"))
                continue
            obj_kind = data.get("kind")
            match obj_kind:
                case "status":
                    status = Status(message=data.get("message"))
                    if status.message == "exit":
                        sys.exit(0)
                    else:
                        print(Error(message=f"Unrecognized status message: {status.message}"))
                case "input":
                    model_input = Input(image=data.get("image"))
                    try:
                        main(model_input.image)
                    except Exception as e:
                        print(Error(message=f"Model crashed while running, skipping the frame: {e}"))
                        continue