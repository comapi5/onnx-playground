import io

import numpy as np
import onnx
import onnxruntime as ort
import PIL
import pytest
import requests
import torch
import torchvision


def download_image():
    # https://www.pakutaso.com/20231006284post-48113.html
    url = "https://user0514.cdnw.net/shared/img/thumb/kotetsuPAR516712026_TP_V.jpg"
    res = requests.get(url)
    image = PIL.Image.open(io.BytesIO(res.content))
    return image


def preprocess_image(image):
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    image = preprocess(image).unsqueeze(0)
    return image


def predict_torch_model(image, model):
    model.eval()
    with torch.no_grad():
        output = model(image)
    predicted_class = torch.max(output, 1)[1]
    return predicted_class.item()


def convert_to_onnx(image, model, onnx_filename):
    torch.onnx.export(model, image, onnx_filename)


def predict_onnx_model(image, model):
    ort_session = ort.InferenceSession(model.SerializeToString())
    ort_inputs = {ort_session.get_inputs()[0].name: image.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    predicted_class = np.argmax(ort_outs[0], axis=1)
    return predicted_class.item()


@pytest.mark.dependency()
def test_resnet18_from_torch():
    # Download and preprocess image
    image = download_image()
    preprocessed_image = preprocess_image(image)

    # Load Torch model
    torch_model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )

    # Predict with Torch model
    torch_output = predict_torch_model(preprocessed_image, torch_model)
    print("Torch model prediction:", torch_output)

    # Convert Torch model to ONNX
    convert_to_onnx(preprocessed_image, torch_model, "resnet18.onnx")

    # Predict with ONNX model
    onnx_model = onnx.load("resnet18.onnx")
    onnx_output = predict_onnx_model(preprocessed_image, onnx_model)
    print("ONNX model prediction:", onnx_output)

    # Compare predictions
    assert torch_output == onnx_output, "Torch and ONNX predictions do not match!"


@pytest.mark.dependency(depends=["test_resnet18_from_torch"])
def test_resnet18_quant():
    onnx_model = onnx.load("resnet18.onnx")


if __name__ == "__main__":
    test_resnet18()
