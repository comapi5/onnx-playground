import functools

import numpy as np
import onnx
import onnxruntime
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
        nodes=[
            helper.make_node(
                operator_name.capitalize(), ["input"], ["output"], **attribute
            )
        ],
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


def test_exp():
    _test_element_wise_operator(
        operator_name="Exp",
        np_func=np.exp,
        np_testing_function=functools.partial(np.testing.assert_allclose, rtol=2e-07),
    )
