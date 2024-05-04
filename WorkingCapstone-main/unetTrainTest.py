import cv2
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model

# Function to configure the GPU (if available)
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print("RuntimeError in setting up GPU:", e)

# Function to post-process the predicted mask
def post_process_mask(predicted_mask):
    median_filtered = cv2.medianBlur(predicted_mask, 5)
    _, binary_mask = cv2.threshold(median_filtered, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    num_labels, labels_im = cv2.connectedComponents(closing)
    if num_labels > 1:
        component_areas = [np.sum(labels_im == i) for i in range(1, num_labels)]
        largest_component = np.argmax(component_areas) + 1  # +1 as index 0 is the background
        largest_mask = (labels_im == largest_component).astype(np.uint8)
        closing = largest_mask * 255  # Make the largest component white in the mask
    return closing

# Function to load the UNet model
def load_unet_model(model_path):
    return load_model(model_path)

# Function to predict the mask for an image using the loaded model
def predict_mask(model, image):
    reshaped_image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(reshaped_image)
    return prediction[0, :, :, 0]  # Remove batch dimension and select channel

# Main function to load images and test the model
def main():
    configure_gpu()
    model_dir = "Unet"

    # Assuming the UNet model is saved with the name 'path_segmentation_model_epochs_5_batch_16_val_split_0.2.keras'
    model_filename = "path_segmentation_model_epochs_5_batch_16_val_split_0.2.keras"

    # Construct the full path to the UNet model
    model_path = os.path.join(model_dir, model_filename)

    # Load the UNet model
    model = load_unet_model(model_path)

    test_folder = 'C:/Users/aa/Documents/code/NewCapstoneForGit/WorkingCapstone/resized'
    test_images = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]

    # Create output folders if they don't exist
    output_folders = ['original_images', 'ground_truth_masks', 'predicted_masks', 'processed_masks']
    for folder in output_folders:
        folder_path = os.path.join(test_folder, folder)
        os.makedirs(folder_path, exist_ok=True)

    # Limit to the last 100 images
    start_index = max(len(test_images) - 100, 0)
    test_images = test_images[start_index:]

    for i, filename in enumerate(test_images):
        image_path = os.path.join(test_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        image_resized = cv2.resize(image, (224, 224))  # Ensure the input image size matches the model expectation
        predicted_mask = predict_mask(model, image_resized)
        predicted_mask = (predicted_mask * 255).astype(np.uint8)
        processed_mask = post_process_mask(predicted_mask)

        # Concatenate the images horizontally
        concatenated_image = np.hstack((image, predicted_mask, processed_mask))

        # Save the concatenated image
        cv2.imwrite(os.path.join(test_folder, output_folders[0], f'concatenated_{i}.jpg'), concatenated_image)

        # Optionally visualize the results
        cv2.imshow('Concatenated Image', concatenated_image)
        if cv2.waitKey(1000) & 0xFF == ord('q'):  # Press 'q' to quit the display between images
            break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
