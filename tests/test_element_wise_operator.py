import functools

import numpy as np
import onnx
import onnxruntime
import pytest
from onnx import TensorProto, helper


def _test_element_wise_operator(
    operator_name,
    np_func,
    input_shape=(3, 256, 256),
    input_min=-1.0,
    input_max=1.0,
    attribute={},
    np_testing_function=np.testing.assert_equal,
):
    input = np.random.uniform(input_min, input_max, size=input_shape).astype(np.float32)

    graph = helper.make_graph(
        nodes=[helper.make_node(operator_name, ["input"], ["output"], **attribute)],
        name=f"test_{operator_name.lower()}",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, input.shape)],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, input.shape)
        ],
    )

    model = helper.make_model(graph)

    session = onnxruntime.InferenceSession(model.SerializeToString())

    ort_output = session.run(None, {"input": input})[0]
    np_output = np_func(input, **attribute)

    np_testing_function(ort_output, np_output)


def test_abs():
    _test_element_wise_operator(
        operator_name="Abs",
        np_func=np.abs,
    )


def test_acos():
    _test_element_wise_operator(
        operator_name="Acos",
        np_func=np.arccos,
    )


def test_acosh():
    _test_element_wise_operator(
        operator_name="Acosh",
        input_min=1,
        input_max=10,
        np_func=np.arccosh,
    )


def test_asin():
    _test_element_wise_operator(
        operator_name="Asin",
        np_func=np.arcsin,
    )


def test_asinh():
    _test_element_wise_operator(
        operator_name="Asinh",
        np_func=np.arcsinh,
    )


def test_atan():
    _test_element_wise_operator(
        operator_name="Atan",
        np_func=np.arctan,
    )


def test_atanh():
    _test_element_wise_operator(
        operator_name="Atanh",
        np_func=np.arctanh,
    )


def test_ceil():
    _test_element_wise_operator(
        operator_name="Ceil",
        np_func=np.ceil,
    )


def test_celu():
    def np_celu(x, alpha):
        return np.where(x > 0, x, alpha * (np.exp(x / alpha) - 1))

    _test_element_wise_operator(
        operator_name="Celu",
        attribute={"alpha": 1.0},
        np_func=np_celu,
        np_testing_function=functools.partial(np.testing.assert_allclose, rtol=1e-03),
    )


def test_cos():
    _test_element_wise_operator(
        operator_name="Cos",
        np_func=np.cos,
        np_testing_function=functools.partial(np.testing.assert_allclose, rtol=1e-06),
    )


def test_cosh():
    _test_element_wise_operator(
        operator_name="Cosh",
        np_func=np.cosh,
    )


def test_elu():
    def np_elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    _test_element_wise_operator(
        operator_name="Elu",
        attribute={"alpha": 1.0},
        np_func=np_elu,
    )


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


def test_erf():
    _test_element_wise_operator(
        operator_name="Erf",
        np_func=np_erf,
        np_testing_function=functools.partial(np.testing.assert_allclose, rtol=0.1),
    )


def test_exp():
    _test_element_wise_operator(
        operator_name="Exp",
        np_func=np.exp,
        np_testing_function=functools.partial(np.testing.assert_allclose, rtol=2e-07),
    )


def test_floor():
    _test_element_wise_operator(
        operator_name="Floor",
        np_func=np.floor,
    )


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


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu(approximate):
    _test_element_wise_operator(
        operator_name="Gelu",
        np_func=np_gelu,
        attribute={"approximate": approximate},
        np_testing_function=functools.partial(np.testing.assert_allclose, rtol=1e-06),
    )


def np_hard_sigmoid(x, alpha=0.2, beta=0.5):
    return np.maximum(0, np.minimum(1, alpha * x + beta))


def test_hard_sigmoid():
    _test_element_wise_operator(
        operator_name="HardSigmoid",
        np_func=np_hard_sigmoid,
    )


def test_log():
    _test_element_wise_operator(
        operator_name="Log",
        np_func=np.log,
        input_min=0.01,
        np_testing_function=functools.partial(np.testing.assert_allclose, rtol=1e-06),
    )


def np_softsign(x):
    return x / (1 + np.abs(x))


def test_softsign():
    _test_element_wise_operator(
        operator_name="Softsign",
        np_func=np_softsign,
        np_testing_function=functools.partial(np.testing.assert_allclose, rtol=1e-06),
    )


def test_sqrt():
    _test_element_wise_operator(
        operator_name="Sqrt",
        np_func=np.sqrt,
        input_min=0,
        np_testing_function=functools.partial(np.testing.assert_allclose, rtol=1e-06),
    )


def np_thresholded_relu(x, alpha=1.0):
    return np.where(x > alpha, x, 0)


@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_thresholded_relu(alpha):
    _test_element_wise_operator(
        operator_name="ThresholdedRelu",
        np_func=np_thresholded_relu,
        attribute={"alpha": alpha},
    )


def test_tan():
    _test_element_wise_operator(
        operator_name="Tan",
        np_func=np.tan,
    )


def test_tanh():
    _test_element_wise_operator(
        operator_name="Tanh",
        np_func=np.tanh,
        np_testing_function=functools.partial(np.testing.assert_allclose, rtol=1e-06),
    )
