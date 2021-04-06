from keras.datasets import mnist
from keras import layers, models, utils, callbacks
# from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


import matplotlib.pyplot as plt

# Define hyper parameters
batch_size = 128
num_classes = 10
epochs = 10
# Image dimensions
x_img = y_img = 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
Here data is provided to CNN, so we are not going to flatten it as we did in DNN.

Reshape x_test, x_train to be re-casted as tensors in 4 dimensions

1 in argument: Its grayscale, and channel is 1
"""
x_train = x_train.reshape(x_train.shape[0], x_img, y_img, 1)
x_test = x_test.reshape(x_test.shape[0], x_img, y_img, 1)
input_shape = (x_img, y_img, 1)

# Normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert 1D class arrays to 10D class matrics
y_train = utils.np_utils.to_categorically(y_train, num_classes)
y_test = utils.np_utils.to_categorically(y_test, num_classes)


def baseline_model():
    model = layers.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    # Input shape parameter is needed for the first time as our NN does not about input initially
    # Subsequent models does not require input shape
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(5, 5), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


