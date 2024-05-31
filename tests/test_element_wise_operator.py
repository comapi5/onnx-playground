import functools

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper


def _test_element_wise_operator(
    input_shape,
    operator_name,
    np_func,
    np_testing_function,
):
    input = np.random.uniform(-10, 10, size=input_shape).astype(np.float32)

    graph = helper.make_graph(
        nodes=[helper.make_node(operator_name.capitalize(), ["input"], ["output"])],
        name=f"test_{operator_name.lower()}",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, input.shape)],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, input.shape)
        ],
    )

    model = helper.make_model(graph)

    session = onnxruntime.InferenceSession(model.SerializeToString())

    ort_output = session.run(None, {"input": input})[0]
    np_output = np_func(input)

    np_testing_function(ort_output, np_output)


def test_abs():
    _test_element_wise_operator(
        input_shape=(3, 256, 256),
        operator_name="Abs",
        np_func=np.abs,
        np_testing_function=np.testing.assert_equal,
    )


def test_acos():
    _test_element_wise_operator(
        input_shape=(3, 256, 256),
        operator_name="Acos",
        np_func=np.arccos,
        np_testing_function=np.testing.assert_equal,
    )


def test_exp():
    _test_element_wise_operator(
        input_shape=(3, 256, 256),
        operator_name="Exp",
        np_func=np.exp,
        np_testing_function=functools.partial(np.testing.assert_allclose, rtol=2e-07),
    )
