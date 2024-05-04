import cv2
from threading import Thread
from djitellopy import Tello
import numpy as np
import keras
import tensorflow as tf

# Function to predict bounding boxes using the loaded model
def predict_bounding_boxes(image, model):
    # Resize the image
    resized = cv2.resize(image, (576, 432))
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Reshape
    reshaped = np.reshape(gray, (1, 432, 576, 1))
    # Predict the bounding box using the model
    prediction = model.predict(reshaped)
    # Extract the coordinates from the prediction
    x = int(prediction[0][0])
    y = int(prediction[0][1])
    w = int(prediction[0][2])
    h = int(prediction[0][3])
    return x, y, w, h

# Movement control function
def move_drone(center_x, image_center_x):
    # Define a tolerance range for the center deviation
    tolerance = 20
    # Ensure center_x is an integer
    center_x = int(center_x)
    # Calculate the deviation from the image center
    deviation = center_x - image_center_x
    # Define a sensitivity factor to adjust the movement responsiveness
    sensitivity = 0.1  # Adjust as needed
    # Calculate the speed adjustment based on deviation and sensitivity
    speed_adjustment = int(deviation * sensitivity)
    # Move the drone forward with adjusted speed
    tello.move_forward(1 + speed_adjustment)  # Adjust speed as needed
# Define a function to perform actions based on detected objects
# Initialize Tello drone
tello = Tello()
tello.connect()
tello.get_battery()
keepStreaming = True
tello.streamon()
# Takeoff
tello.takeoff()
# Load the saved model
model = keras.models.load_model('15e8b1v.h5')
# Function to stream video from Tello and perform inference
def videoStreamer():
    while keepStreaming:
        try:
            frame = tello.get_frame_read().frame

            # Perform inference using the loaded model
            if model is not None:
                # Predict bounding boxes and get the center of the line
                x, _, w, _ = predict_bounding_boxes(frame, model)
                line_center_x = x + (w // 2)  # Calculate the center of the line

                # Move the drone based on the detected line's center
                move_drone(line_center_x, frame.shape[1] // 2)  # Pass the image center as a reference

                # Draw indicators on the frame
                cv2.rectangle(frame, (x, 0), (x + w, frame.shape[0]), (0, 255, 0), 2)  # Draw bounding box around the line
                cv2.circle(frame, (line_center_x, frame.shape[0] // 2), 5, (0, 0, 255), -1)  # Draw center of the line

            # Display the frame with bounding boxes and indicators
            cv2.imshow('Tello Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
        except Exception as e:
            print("Error during streaming:", e)
            continue

    cv2.destroyAllWindows()

# Run the streamer in a separate thread
streamer = Thread(target=videoStreamer)
streamer.start()

# Your control logic based on the inference result can go here
# Example: Based on the detected objects, control the drone to perform specific actions

# Join the streamer thread
streamer.join()
