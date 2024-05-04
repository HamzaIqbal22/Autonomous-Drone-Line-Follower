import cv2
import numpy as np
import keras
from djitellopy import Tello

# Load the saved model
model = keras.models.load_model('5e_16b_2v.h5')

def process_frame(frame):
    resized = cv2.resize(frame, (224, 224))  # Resize the image to match the model's expected input
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    reshaped = np.reshape(gray, (1, 224, 224, 1))  # Reshape the gray image

    prediction = model.predict(reshaped)
    x, y, w, h = map(int, prediction[0])

    # Draw the bounding box on the resized image
    cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return resized

def capture_and_display():
    # Connect to the Tello drone
    tello = Tello()
    tello.connect()
    tello.streamon()

    frame_read = tello.get_frame_read()

    try:
        while True:
            frame = frame_read.frame
            if frame is None:
                print("Failed to capture image from drone's camera")
                continue

            processed_frame = process_frame(frame)

            # Display the processed frame
            cv2.imshow("Processed Frame", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
                break
    finally:
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()

capture_and_display()
