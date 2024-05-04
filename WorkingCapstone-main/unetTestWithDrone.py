import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from djitellopy import Tello

# Function to configure the GPU (if available)
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("RuntimeError in setting up GPU:", e)

# Function to post-process the predicted mask
def post_process_mask(predicted_mask):
    """ Process the predicted mask to refine the contours. """
    median_filtered = cv2.medianBlur(predicted_mask, 5)
    _, binary_mask = cv2.threshold(median_filtered, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    num_labels, labels_im = cv2.connectedComponents(closing)
    largest_mask = np.zeros_like(binary_mask)  # Initialize with zeros
    if num_labels > 1:
        component_areas = [(labels_im == i).sum() for i in range(1, num_labels)]
        largest_component = 1 + np.argmax(component_areas)
        largest_mask = np.uint8(labels_im == largest_component) * 255
    return largest_mask

# Function to load the model
def load_unet_model(model_path):
    return load_model(model_path)

# Function to predict the mask for an image using the loaded model
def predict_mask(model, image):
    reshaped_image = np.expand_dims(image, axis=0)
    prediction = model.predict(reshaped_image)
    return prediction[0, :, :, 0]

def main():
    configure_gpu()
    model_path = 'path_segmentation_model_epochs_5_batch_16_val_split_0.2.keras'
    model = load_unet_model(model_path)

    # Connect to the drone and start video stream
    drone = Tello()
    drone.connect()
    print(f"Battery level: {drone.get_battery()}%")
    drone.streamon()

    try:
        frame_read = drone.get_frame_read()
        while True:
            frame = frame_read.frame
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized_image = cv2.resize(gray, (224, 224))  # Resize image to match model input
                predicted_mask = predict_mask(model, resized_image)
                predicted_mask = (predicted_mask * 255).astype(np.uint8)  # Convert to uint8 image
                processed_mask = post_process_mask(predicted_mask)

                # Display the processed mask
                cv2.imshow('Processed Mask', processed_mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        drone.streamoff()
        drone.end()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
