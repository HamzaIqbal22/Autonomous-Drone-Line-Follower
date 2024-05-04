import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard
import random
import keras

import time
import datetime

print(tf.config.list_physical_devices('GPU'))

# Add GPU memory growth option
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Function to resize images
def resize_images(input_folder, output_folder, target_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)
    for file in files:
        img_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        try:
            # Check if the output folder already contains the resized image
            if os.path.exists(output_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                raise Exception("Failed to read image")

            resized_img = cv2.resize(img, target_size)
            cv2.imwrite(output_path, resized_img)
        except Exception as e:
            print(f"Error processing image: {img_path}")
            print(f"Error details: {e}")

# Function to augment images
def augment_images(input_folder, output_folder, target_size, num_augmented_per_image):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)
    for file in files:
        img_path = os.path.join(input_folder, file)
        try:
            # Check if the output folder already contains the expected number of augmented images for this input image
            existing_augmented_images = [name for name in os.listdir(output_folder) if name.startswith(file.split('.')[0])]
            if len(existing_augmented_images) >= num_augmented_per_image:
                continue

            img = cv2.imread(img_path)
            if img is None:
                raise Exception("Failed to read image")

            for i in range(num_augmented_per_image - len(existing_augmented_images)):
                augmented_img = augment_image(img, target_size)
                output_path = os.path.join(output_folder, f"{file.split('.')[0]}_{i + len(existing_augmented_images)}.jpg")
                cv2.imwrite(output_path, augmented_img)
        except Exception as e:
            print(f"Error processing image: {img_path}")
            print(f"Error details: {e}")

def augment_image(image, target_size):
    # Horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Zooming out and rotation (randomly applied)
    if random.random() > 0.5:
        scale_factor = random.uniform(0.7, 1.3)  # Adjust the range for zooming out
        angle = random.uniform(-10, 10)  # Adjust the range for rotation
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
        image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    # Resize to target size
    augmented_image = cv2.resize(image, target_size)

    return augmented_image

# Function to create and train the VGG16-based model
def train_model(images, labels, epoch, batch, filelog, val_split):
    print("\n")
    print("Starting test: " + filelog)

    # Load pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(432, 576, 3))

    # Freeze VGG16 layers
    base_model.trainable = False

    # Create model
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),  # Reduced units
        Dropout(0.5),
        Dense(4, activation='linear')  # 4 outputs for x, y, w, and h
    ])

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # Early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Model checkpoint callback
    checkpoint_path = f"{filelog}_best_model.keras"
    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')

    # CSV logger callback
    filename = filelog + "vgg16"+ ".csv"
    history_logger = CSVLogger(filename, separator=",", append=True)

    # Learning rate scheduler callback
    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    # Train the model
    history = model.fit(images, labels, epochs=epoch, batch_size=batch, validation_split=val_split,
                        callbacks=[early_stop, model_checkpoint, history_logger, lr_callback])

    # Save the trained model
    model.save(filelog + "vgg16" + '.keras')
    print("Finished test: " + filelog)

    return history

# Function to plot training history
def plot_training_history(history, file_prefix):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(f'{file_prefix}_accuracy.png')

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{file_prefix}_loss.png')

if __name__ == "__main__":
    # Set paths and parameters
    input_folder = "C:/Users/aa/Documents/code/NewCapstoneForGit/WorkingCapstone/photos"
    output_folder = "C:/Users/aa/Documents/code/NewCapstoneForGit/WorkingCapstone/resized/"
    augmented_output_folder = "C:/Users/aa/Documents/code/NewCapstoneForGit/WorkingCapstone/augmented/"
    target_size = (576, 432)  # Decreased size
    num_images = 1000
    image_folder = "resized"
    epoch = 15
    batch = 8  # Reduced batch size
    val_split = 0.1
    filelog = "15e8b1v"  # Update accordingly
    num_augmented_per_image = 1  # Number of augmented images to generate per original image

    try:
        # Resize images
        resize_images(input_folder, output_folder, target_size)

        # Augment images
        augment_images(output_folder, augmented_output_folder, target_size, num_augmented_per_image)

        # Load resized images and labels
        images = []
        labels = []
        for i in range(num_images):
            image_path = f"{output_folder}photo_{i + 1}.jpg"
            img = cv2.imread(image_path)
            if img is None:
                raise Exception(f"Failed to read image: {image_path}")

            edges = cv2.Canny(img, 100, 200)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(edges, kernel)
            contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            longest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            is_closed = False
            total = []
            rectangles = []
            if len(longest_contours) >= 2:
                for contour in longest_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    rectangles.append((x, y, w, h))

                if rectangles:
                    x1, y1, w1, h1 = rectangles[0]
                    x2, y2, w2, h2 = rectangles[1]

                x = min(x1, x2)
                y = min(y1, y2)
                w = max(x1 + w1, x2 + w2) - x
                h = max(y1 + h1, y2 + h2) - y

                labels.append(np.array([x, y, w, h]))
            elif len(longest_contours) == 1:
                for contour in longest_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    labels.append(np.array([x, y, w, h]))
            else:
                labels.append(np.array([-1, -1, -1, -1]))

            images.append(img)  # Append original image

        # Load augmented images and labels
        for file in os.listdir(augmented_output_folder):
            image_path = os.path.join(augmented_output_folder, file)
            img = cv2.imread(image_path)
            if img is None:
                raise Exception(f"Failed to read image: {image_path}")

            edges = cv2.Canny(img, 100, 200)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(edges, kernel)
            contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            longest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            is_closed = False
            total = []
            rectangles = []
            if len(longest_contours) >= 2:
                for contour in longest_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    rectangles.append((x, y, w, h))

                if rectangles:
                    x1, y1, w1, h1 = rectangles[0]
                    x2, y2, w2, h2 = rectangles[1]

                x = min(x1, x2)
                y = min(y1, y2)
                w = max(x1 + w1, x2 + w2) - x
                h = max(y1 + h1, y2 + h2) - y

                labels.append(np.array([x, y, w, h]))
            elif len(longest_contours) == 1:
                for contour in longest_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    labels.append(np.array([x, y, w, h]))
            else:
                labels.append(np.array([-1, -1, -1, -1]))

            images.append(img)  # Append augmented image

        images = np.array(images)
        labels = np.array(labels)

        # Train model
        history = train_model(images, labels, epoch, batch, filelog, val_split)

        # Plot training history
        plot_training_history(history, filelog)

    except Exception as e:
        print(f"An error occurred: {e}")
