import cv2
import numpy as np
import os
import itertools
from itertools import product
from datetime import datetime
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from unetmodel import unet_model
import pandas as pd

def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("RuntimeError in setting up GPU:", e)

def create_mask_for_image(image):
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image is in color
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # The image is already in grayscale
    edges = cv2.Canny(gray, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    return mask / 255.0  # Normalize to 0-1

configure_gpu()
input_folder = 'C:/Users/aa/Documents/code/NewCapstoneForGit/WorkingCapstone/resized'
num_images = 1730
images, masks = [], []

for i in range(num_images):
    image_path = os.path.join(input_folder, f"photo_{i + 1}.jpg")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to read image: {image_path}")
        continue
    img = cv2.resize(img, (224, 224))
    mask = create_mask_for_image(img)
    images.append(img[..., np.newaxis])
    masks.append(mask[..., np.newaxis])

images = np.array(images, dtype=np.float32) / 255.0
masks = np.array(masks, dtype=np.float32)
# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Apply augmentation and reapply mask logic to augmented images
augmented_images = []
augmented_masks = []
total_augmented = 0
for img, mask in zip(images, masks):
    img_batch = np.expand_dims(img, axis=0)
    mask_batch = np.expand_dims(mask, axis=0)
    aug_iter = datagen.flow(img_batch, mask_batch, batch_size=1)
    for _ in range(5):
        aug_img, aug_mask = next(aug_iter)
        augmented_images.append(aug_img[0])
        augmented_masks.append(aug_mask[0])
        total_augmented += 1

augmented_images = np.array(augmented_images)
augmented_masks = np.array(augmented_masks)

# Combine original and augmented datasets
all_images = np.concatenate([images, augmented_images])
all_masks = np.concatenate([masks, augmented_masks])

## Define the combinations of epochs, batch sizes, and validation splits
epochs_list = [5] # 5 10 15 (do 10160.1/0.2) do 158/160.2/0.3
batch_sizes = [16] # 8 16 (32 is too much)
val_splits = [0.3] # 0.1 0.2 0.3

# Iterate through all combinations
for epochs, batch_size, val_split in product(epochs_list, batch_sizes, val_splits):
    # Model training
    model = unet_model()
    history = model.fit(all_images, all_masks, epochs=epochs, batch_size=batch_size, validation_split=val_split)

    # Save the model with a distinguishable name
    model_name = f'path_segmentation_model_epochs_{epochs}_batch_{batch_size}_val_split_{val_split}.keras'
    model.save(model_name)

    # Save training history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'history_{model_name}.csv', index=False)

    # Clear session to release memory
    tf.keras.backend.clear_session()