import pathlib
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from yolov5 import export


def test_yolov5s_to_onnx():
    torch_model_path = pathlib.Path("yolov5s.pt")
    onnx_model_path = pathlib.Path("yolov5s.onnx")

    torch.hub.load("ultralytics/yolov5", "custom", str(torch_model_path))
    export.run(weights=str(torch_model_path), include=["onnx"])

    torch_model_path.unlink()
    onnx_model_path.unlink()
