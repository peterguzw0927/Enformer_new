import onnxruntime as ort
import numpy as np

# Load the ONNX model
model_path = 'test_enformer.onnx'
session = ort.InferenceSession(model_path)

# Get the names of the input and output nodes
input_names = [input.name for input in session.get_inputs()]
output_names = [output.name for output in session.get_outputs()]

print(f"Input names: {input_names}")
print(f"Output names: {output_names}")

# Prepare input data (this will depend on your model's input requirements)
# Example: Create a dummy input array
# Make sure the input shape matches the model's expected input shape
dummy_input = np.random.rand(1, 196_608, 4).astype(np.float32)  # Example shape for an image

# Run inference
results = session.run(output_names, {input_names[0]: dummy_input})

# Process results
print(f"Output results: {results}")

