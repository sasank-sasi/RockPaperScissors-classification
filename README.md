#Rock Paper Scissors Classification

Overview

This project involves training a machine learning model to classify images of hand gestures representing Rock, Paper, and Scissors. The model is based on a convolutional neural network (CNN) and is capable of predicting the gesture shown in a given image.

Features

Image Classification: The model predicts whether an input image represents Rock, Paper, or Scissors.
Real-time Webcam Prediction: Allows users to test the model by capturing real-time hand gestures through their webcam.
Confidence Score: Provides a confidence score for each prediction.
Technologies Used

Python
Keras (with TensorFlow backend)
OpenCV
Matplotlib
Joblib (for model persistence)
Usage

Training the Model:
Train the model using the provided dataset.
Execute the training script:
bash
Copy code
python train_model.py
Real-time Webcam Prediction:
Run the script for real-time prediction:
bash
Copy code
python webcam_prediction.py
Capture hand gestures through your webcam and see real-time predictions.
Single Image Prediction:
Use the trained model to predict gestures from individual images:
bash
Copy code
python predict_single_image.py -i path/to/image.jpg
Replace path/to/image.jpg with the path to the image you want to classify.
Project Structure

bash
Copy code
.
├── data/               # Training dataset (Rock, Paper, Scissors images)
├── models/             # Directory to store the trained model
├── src/
│   ├── train_model.py  # Script to train the classification model
│   ├── webcam_prediction.py  # Real-time webcam prediction script
│   ├── predict_single_image.py  # Prediction script for a single image
├── requirements.txt    # Dependencies required for the project
├── README.md           # Project overview, instructions, and documentation
Installation

Clone the repository:
bash
Copy code
git clone https://github.com/your-username/RockPaperScissors-classification.git
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Run the project following the usage instructions.

License

This project is licensed under the MIT License.

