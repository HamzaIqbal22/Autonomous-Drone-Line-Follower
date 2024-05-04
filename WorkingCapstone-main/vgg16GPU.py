import itertools
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.utils.vis_utils import plot_model
# Check available physical GPUs and memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def train_model(images, labels, epochs, batches, val_splits):
    for epoch, batch, val_split in itertools.product(epochs, batches, val_splits):
        filelog = f"{epoch}e{batch}b{int(val_split * 10)}v"
        print("\nStarting test: " + filelog)
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False

        model = Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(224, 224, 3)),
            base_model,
            Flatten(),
            BatchNormalization(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),
            Dense(4, activation='linear')
        ])

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                      loss='mean_squared_error',
                      metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint_path = f"{filelog}_best_model.keras"
        model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
        filename = filelog + ".csv"
        history_logger = CSVLogger(filename, separator=",", append=True)

        history = model.fit(images, labels, epochs=epoch, batch_size=batch, validation_split=val_split,
                            callbacks=[early_stop, model_checkpoint, history_logger])
        model.save(filelog + '.keras')
        plot_model(model, to_file='model_plot_Vgg16GPU.png', show_shapes=True, show_layer_names=True)
        print("Finished test: " + filelog)
        plot_training_history(history, filelog)
        # Clear GPU memory
        tf.keras.backend.clear_session()

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
    input_folder = "C:/Users/aa/Documents/code/NewCapstoneForGit/WorkingCapstone/resized/"
    target_size = (224, 224)
    num_images = 1730
    epochs = [10]  #5,10,15 List of epochs for each test
    batches = [8]  # 8,16,32List of batch sizes for each test
    val_splits = [0.1,0.2, 0.3]  # List of validation splits for each test
    # Initialize an empty list to store test names
    filelogs = []
    num_augmented_per_image = 5

    try:
        images = []
        labels = []
        for i in range(num_images):
            image_path = os.path.join(input_folder, f"photo_{i + 1}.jpg")
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

            images.append(img)

        images = np.array(images)
        labels = np.array(labels)

        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )

        augmented_images = []
        augmented_labels = []

        # Generate augmented images and labels
        for i in range(num_images):
            img = images[i]
            img_batch = np.expand_dims(img, axis=0)
            for j in range(num_augmented_per_image):
                augmented_img = datagen.flow(img_batch, batch_size=1, shuffle=False).next()[0].astype(np.uint8)
                augmented_edges = cv2.Canny(augmented_img, 100, 200)
                augmented_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                augmented_dilated = cv2.dilate(augmented_edges, augmented_kernel)
                augmented_contours, _ = cv2.findContours(augmented_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                augmented_longest_contours = sorted(augmented_contours, key=cv2.contourArea, reverse=True)[:2]
                if len(augmented_longest_contours) >= 2:
                    augmented_rectangles = []
                    for contour in augmented_longest_contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        augmented_rectangles.append((x, y, w, h))
                    if augmented_rectangles:
                        x1, y1, w1, h1 = augmented_rectangles[0]
                        x2, y2, w2, h2 = augmented_rectangles[1]
                        x = min(x1, x2)
                        y = min(y1, y2)
                        w = max(x1 + w1, x2 + w2) - x
                        h = max(y1 + h1, y2 + h2) - y
                        augmented_labels.append(np.array([x, y, w, h]))
                    else:
                        augmented_labels.append(np.array([-1, -1, -1, -1]))
                elif len(augmented_longest_contours) == 1:
                    for contour in augmented_longest_contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        augmented_labels.append(np.array([x, y, w, h]))
                else:
                    augmented_labels.append(np.array([-1, -1, -1, -1]))

                augmented_images.append(augmented_img)

        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)

        all_images = np.concatenate((images, augmented_images), axis=0)
        all_labels = np.concatenate((labels, augmented_labels), axis=0)

        train_model(all_images, all_labels, epochs, batches, val_splits)

    except Exception as e:
        print(f"An error occurred: {e}")
