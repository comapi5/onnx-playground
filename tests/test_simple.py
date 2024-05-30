import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper


def test_abs():
    input = np.random.uniform(-100, 100, size=(3, 256, 256)).astype(np.float32)

    graph = helper.make_graph(
        nodes=[helper.make_node("Abs", ["input"], ["output"])],
        name="test_abs",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, input.shape)],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, input.shape)
        ],
    )

    model = helper.make_model(graph)

    session = onnxruntime.InferenceSession(model.SerializeToString())

    ort_output = session.run(None, {"input": input})[0]
    np_output = np.abs(input)

    np.testing.assert_equal(ort_output, np_output)


def test_exp():
    input = np.random.uniform(-1, 1, size=(3, 256, 256)).astype(np.float32)

    graph = helper.make_graph(
        nodes=[helper.make_node("Exp", ["input"], ["output"])],
        name="test_exp",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, input.shape)],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, input.shape)
        ],
    )

    model = helper.make_model(graph)

    session = onnxruntime.InferenceSession(model.SerializeToString())

    ort_output = session.run(None, {"input": input})[0]
    np_output = np.exp(input)

    np.testing.assert_allclose(ort_output, np_output, rtol=2e-07)
