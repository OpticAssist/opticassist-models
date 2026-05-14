import base64
import torch
from ultralytics import YOLO
import sys



def main(img):
    print("hello, world!")
    decoded_bytes = base64.b64decode(img)
    model = YOLO("yolo26n.pt")
    results = model.train(data="img",epochs=3)
    print(decoded_bytes)

if __name__ == '__main__':
    if(len(sys.argv)>1):
        main(sys.argv[1])
    else:
        ...
