import onnx

onnx_model = onnx.load("test_enformer.onnx")

onnx_model.graph.input[0]
