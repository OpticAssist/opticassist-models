#!/usr/bin/env python3

import subprocess

# get the onnx file
subprocess.run(
    [
     "yolo",
     "export",
     "model=yolo26n.pt",
     "format=onnx",
     "opset=13",
     "simplify=True"
     ], check=True
)

# delete downloaded pt file
subprocess.run(
 [
  "rm",
  "./yolo26n.pt"
 ], check=True
)





