import cv2
import numpy as np
import onnxruntime

# Load the ONNX model
onnx_model = onnxruntime.InferenceSession("modnet.onnx")
input_name = onnx_model.get_inputs()[0].name
output_name = onnx_model.get_outputs()[0].name

# Load and preprocess the input image
image = cv2.imread("image1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (512, 512))
input_tensor = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
input_tensor = np.expand_dims(input_tensor, axis=0)

# Run inference
try:
    outputs = onnx_model.run([output_name], {input_name: input_tensor})
    matte = outputs[0][0][0]
except Exception as e:
    print(f"Error during model inference: {str(e)}")
    raise

# Post-process the output
matte = cv2.resize(matte, (image.shape[1], image.shape[0]))
foreground = image * matte[:, :, np.newaxis]
background = image * (1 - matte[:, :, np.newaxis])

# Save results
cv2.imwrite("output_matte.png", matte * 255)
cv2.imwrite("output_foreground.png", foreground)
cv2.imwrite("output_background.png", background)
