import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model

# Load the model
loaded_model = load_model('/Users/sasanksasi/Downloads/ANN-LAB2/rps.h5')

# Open a connection to the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture a single frame
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow("Camera Feed", frame)

    # Check if the user presses 'c' to capture a frame
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # Process the captured frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

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
        print(f'Predicted Class: {pred_class_name}')
        print(f'Confidence Score: {prediction_result[0][x] * 100}')

        # Break the loop after capturing a frame
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
