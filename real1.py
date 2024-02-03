import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
loaded_model = load_model('/Users/sasanksasi/Downloads/ANN-LAB2/rps.h5')

# Open a connection to the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Function to process and classify hand gestures
def classify_gesture(hand_roi):
    # Resize the hand region of interest (ROI) to match the model input size
    img = cv2.resize(hand_roi, (64, 64), interpolation=cv2.INTER_AREA)

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

    return pred_class_name, prediction_result[0][x] * 100

while True:
    # Capture a single frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use a skin color range to detect hands
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(frame, lower_skin, upper_skin)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours
    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw the bounding box around the detected hand
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the hand region of interest (ROI)
        hand_roi = frame[y:y+h, x:x+w]

        # Classify the hand gesture
        pred_class, confidence = classify_gesture(hand_roi)

        # Display the predicted gesture
        cv2.putText(frame, f'Gesture: {pred_class}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f'Confidence: {confidence:.2f}%', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the processed frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
