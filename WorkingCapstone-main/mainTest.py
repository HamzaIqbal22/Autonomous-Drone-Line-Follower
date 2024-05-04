import cv2
import numpy as np
import keras
import os

# Assuming the model is saved in a folder named "Main"
model_dir = "Main"

# Check if the model directory exists
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

# Define the filename of the model
model_filename = "5e_16b_2v.h5"

# Construct the full path to the model
model_path = os.path.join(model_dir, model_filename)

# Load the model
model = keras.models.load_model(model_path)

def predict_bounding_boxes(image_path, output_folder):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return

    resized = cv2.resize(image, (224, 224))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    reshaped = np.reshape(gray, (1, 224, 224, 1))  # Reshape the resized image

    prediction = model.predict(reshaped)
    x, y, w, h = map(int, prediction[0])

    cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the labeled image
    cv2.imshow("Labeled Image", resized)
    cv2.waitKey(200)  # Wait indefinitely for a key press

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image with bounding box to the specified folder
    output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_image_path, resized)

# Set the output folder
output_folder = 'maintestimages'

for i in range(1630, 1730):
    image_path = f"resized/photo_{i}.jpg"
    predict_bounding_boxes(image_path, output_folder)
