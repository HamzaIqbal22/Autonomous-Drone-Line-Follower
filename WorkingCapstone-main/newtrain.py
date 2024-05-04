import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization, InputLayer
from keras.callbacks import EarlyStopping, CSVLogger, TensorBoard
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
def makemodel(images, labels, epoch, batch, filelog, val_split, tensorboard_callback=None):
    print("\n")
    print("Starting test: " + filelog)
    model = Sequential()
    model.add(InputLayer(input_shape=(224, 224, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='linear'))  # 4 outputs for x, y, w, and h

    # Compile the model
    model.compile(optimizer=Adam(),
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    filename = filelog + ".csv"
    history_logger = CSVLogger(filename, separator=",", append=True)

    callbacks = [early_stop, history_logger]
    if tensorboard_callback:
        callbacks.append(tensorboard_callback)

    # Train the model
    model.fit(images, labels, epochs=epoch, batch_size=batch,
              validation_split=val_split, callbacks=callbacks)

    model.save(filelog + '.h5')
    plot_model(model, to_file='model_plot_mainGPU.png', show_shapes=True, show_layer_names=True)
    print("Finished test: " + filelog)
    tf.keras.backend.clear_session()
