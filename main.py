import sys
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import json

def main(img_base64: str):
    try:
        decoded_bytes = base64.b64decode(img_base64)

        np_arr = np.frombuffer(decoded_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        model = YOLO("yolov8s-cls.pt")

        results = model(img)



if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
