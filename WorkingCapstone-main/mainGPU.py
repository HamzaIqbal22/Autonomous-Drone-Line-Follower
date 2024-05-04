import cv2
import numpy as np
import os
import itertools
from datetime import datetime
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from newtrain import makemodel

def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("RuntimeError in setting up GPU:", e)

def process_and_label_image(image):
    edges = cv2.Canny(image, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [cv2.boundingRect(c) for c in sorted(contours, key=cv2.contourArea, reverse=True)[:2]]

    if len(rectangles) >= 2:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = rectangles
        label = np.array([min(x1, x2), min(y1, y2), max(x1 + w1, x2 + w2) - min(x1, x2), max(y1 + h1, y2 + h2) - min(y1, y2)])
    elif rectangles:
        label = np.array(rectangles[0])
    else:
        label = np.array([-1, -1, -1, -1])
    return label

configure_gpu()
input_folder = 'C:/Users/aa/Documents/code/NewCapstoneForGit/WorkingCapstone/resized'
num_images = 1730
images, labels = [], []

for i in range(num_images):
    image_path = os.path.join(input_folder, f"photo_{i + 1}.jpg")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to read image: {image_path}")
        continue
    img = cv2.resize(img, (224, 224))
    label = process_and_label_image(img)
    images.append(img[..., np.newaxis])  # Add channel dimension
    labels.append(label)

images = np.array(images, dtype=np.float32) / 255.0  # Normalize images
labels = np.array(labels, dtype=np.float32)

# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Apply augmentation and reapply label logic to augmented images
augmented_images = []
augmented_labels = []
total_augmented = 0  # Initialize a counter for augmented images
for img in images:
    img_batch = np.expand_dims(img, axis=0)  # Add batch dimension
    for _ in range(5):  # Define the number of augmented images per original image
        aug_img = datagen.flow(img_batch, batch_size=1).next()[0]
        aug_img_uint8 = (aug_img * 255).astype(np.uint8)  # Convert back to uint8
        aug_label = process_and_label_image(aug_img_uint8[:, :, 0])  # Reapply label extraction to augmented image
        augmented_images.append(aug_img)
        augmented_labels.append(aug_label)
        total_augmented += 1  # Increment the counter

augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Combine original and augmented datasets
all_images = np.concatenate([images, augmented_images])
all_labels = np.concatenate([labels, augmented_labels])

# Training configurations
epochs = [5]  # Example values
batches = [16]  # Example values
val_splits = [0.2]  # Example values

# Train the model using makemodel
for epoch, batch, val_split in itertools.product(epochs, batches, val_splits):
    test_name = f"{epoch}e_{batch}b_{int(val_split * 10)}v"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"logs/{test_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    makemodel(all_images, all_labels, epoch, batch, test_name, val_split, tensorboard_callback)

