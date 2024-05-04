import cv2
import numpy as np
import keras
from djitellopy import Tello

# Load the saved model
model = keras.models.load_model('5e16b2v.keras')

def process_frame(frame):
    resized = cv2.resize(frame, (224, 224))
    reshaped = np.reshape(resized, (1, 224, 224, 3))  # Reshape the resized image

    prediction = model.predict(reshaped)
    x, y, w, h = map(int, prediction[0])

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
