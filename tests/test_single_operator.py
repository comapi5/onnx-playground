import numpy as np
import onnx
import onnxruntime
import pytest
from onnx import TensorProto, helper


def np_celu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x / alpha) - 1))


def np_elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def np_erf(x):
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911

    sign = np.sign(x)
    x = np.abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return sign * y


def np_gelu(x, approximate="none"):
    if approximate == "none":
        return 0.5 * x * (1 + np_erf(x / np.sqrt(2)))
    elif approximate == "tanh":
        return (
            0.5
            * x
            * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        )
    else:
        raise ValueError("Invalid value for 'approximate'. Choose 'none' or 'tanh'.")


def np_hard_sigmoid(x, alpha=0.2, beta=0.5):
    return np.maximum(0, np.minimum(1, alpha * x + beta))


def np_mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))


def np_relu(x):
    return np.maximum(0, x)


def np_softplus(x):
    return np.log1p(np.exp(x))


def np_softsign(x):
    return x / (1 + np.abs(x))


def np_thresholded_relu(x, alpha=1.0):
    return np.where(x > alpha, x, 0)


@pytest.mark.parametrize(
    "operator_name, np_function, input_min, input_max, attribute",
    [
        # default attributes
        ("Abs", np.abs, -5, 5, {}),
        ("Acos", np.arccos, -1, 1, {}),  # -1 <= x <= 1
        ("Acosh", np.arccosh, 1, 5, {}),  # 1 <= x < inf
        ("Asin", np.arcsin, -1, 1, {}),  # -1 <= x <= 1
        ("Asinh", np.arcsinh, -5, 5, {}),
        ("Atan", np.arctan, -5, 5, {}),
        ("Atanh", np.arctanh, -1, 1, {}),  # -1 <= x <= 1
        ("Ceil", np.ceil, -5, 5, {}),
        ("Celu", np_celu, -5, 5, {}),
        ("Cos", np.cos, -5, 5, {}),
        ("Cosh", np.cosh, -1, 1, {}),  # -1 <= x <= 1
        ("Elu", np_elu, -5, 5, {}),
        ("Erf", np_erf, -5, 5, {}),
        ("Exp", np.exp, -5, 5, {}),
        ("Floor", np.floor, -5, 5, {}),
        ("Gelu", np_gelu, -5, 5, {}),
        ("HardSigmoid", np_hard_sigmoid, -5, 5, {}),
        ("Log", np.log, 0.01, 5, {}),  # 0 < x < inf
        ("Mish", np_mish, -5, 5, {}),
        ("Relu", np_relu, -5, 5, {}),
        ("Sign", np.sign, -5, 5, {}),
        ("Sin", np.sin, -5, 5, {}),
        ("Sinh", np.sinh, -1, 1, {}),  # -1 <= x <= 1
        ("Softplus", np_softplus, -5, 5, {}),
        ("Softsign", np_softsign, -5, 5, {}),
        ("Sqrt", np.sqrt, 0, 5, {}),  # 0 <= x < inf
        ("ThresholdedRelu", np_thresholded_relu, -5, 5, {}),
        ("Tan", np.tan, -5, 5, {}),
        ("Tanh", np.tanh, -5, 5, {}),
    ],
)
def test_element_wise_operator(
    operator_name, np_function, input_min, input_max, attribute
):
    input_shape = (3, 256, 256)
    inp = np.random.uniform(input_min, input_max, size=input_shape).astype(np.float32)

    graph = helper.make_graph(
        nodes=[helper.make_node(operator_name, ["input"], ["output"], **attribute)],
        name=f"test_{operator_name.lower()}",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)
        ],
    )
    model = helper.make_model(graph)

    session = onnxruntime.InferenceSession(model.SerializeToString())

    ort_output = session.run(None, {"input": inp})[0]
    np_output = np_function(inp, **attribute)

    np.testing.assert_allclose(ort_output, np_output, rtol=0.1)
