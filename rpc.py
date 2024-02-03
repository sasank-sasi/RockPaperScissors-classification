import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model

# Load the model
loaded_model = load_model('/Users/sasanksasi/Downloads/ANN-LAB2/rps.h5')

# Prompt the user for the image path
user_image_path = '/Users/sasanksasi/Downloads/ANN-LAB2/t4.png'

# Read the user-provided image
img = plt.imread(user_image_path)
img = np.array(img)

# Check if the image has an alpha channel (transparency)
if img.shape[2] == 4:
    # Remove the alpha channel
    img = img[:, :, :3]

# Ensure the image has 3 channels (for RGB images)
if len(img.shape) == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# Resize the image to (64, 64)
img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

# Check the shape after resizing
if img.shape != (64, 64, 3):
    raise ValueError(f"Resized image has unexpected shape: {img.shape}")

# Reshape to (1, 12288)
inp = np.reshape(img, (1, 12288))

# Make predictions using the loaded model
prediction_result = loaded_model.predict(inp)
x = np.argmax(prediction_result)

# Map class index to class name
if x == 0:
    pred_class_name = 'paper'
elif x == 1:
    pred_class_name = 'rock'
else:
    pred_class_name = 'scissors'

# Display the result
print(f'Test Image: {user_image_path}')
print(f'Predicted Class: {pred_class_name}')
print(f'Confidence Score: {prediction_result[0][x] * 100}')
