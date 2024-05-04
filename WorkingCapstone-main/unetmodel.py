from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from keras.models import Model
from keras.utils.vis_utils import plot_model# Define the U-Net model with dropout for regularization
def unet_model(input_size=(224, 224, 1)):
    inputs = Input(input_size)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)  # Dropout for regularization
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Dropout(0.1)(c2)  # Dropout for regularization
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)

    u1 = UpSampling2D((2, 2))(c3)
    concat1 = Concatenate()([u1, c2])
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat1)
    u2 = UpSampling2D((2, 2))(c4)
    concat2 = Concatenate()([u2, c1])
    c5 = Conv2D(16, (3, 3), activation='relu', padding='same')(concat2)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='model_plot_UNET.png', show_shapes=True, show_layer_names=True)
    return model
