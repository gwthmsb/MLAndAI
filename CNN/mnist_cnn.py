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
y_train = utils.np_utils.to_categorical(y_train, num_classes)
y_test = utils.np_utils.to_categorical(y_test, num_classes)


def baseline_model():
    model = models.Sequential()
    # Trainable Parameters count = (# of filters in conv layer) * (# neurons +1)
    # # neurons = kernel size parameters (Even kernel depth is considered, here it is 1)
    # Here, count = 32 * 26 = 832
    model.add(layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    # Input shape parameter is needed for the first time as our NN does not about input initially
    # Subsequent models does not require input shape
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # Count = (5*5*32+1)*64 = 51264
    model.add(layers.Conv2D(64, kernel_size=(5, 5), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    # No of parameters for dense layer: (# neurons in prev layer +1 ) * (# neurons in dense layer)
    # After flatten: size would(neurons) be 1024
    # Neuron count = (1024+1)*64 = 65600
    model.add(layers.Dense(64, activation='relu'))
    # Neuron count = (64+1)*10 = 650
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model

def baseline_model_fit(model):
    model_log = model.fit(x_train,  y_train, epochs=epochs, batch_size=batch_size, verbose=1,
              validation_data=(x_test, y_test))
    return model_log

def run_basline_model():
    model = baseline_model()
    model_log = baseline_model_fit(model)
    evaluate_model(model_log)


def smaller_model()->models.Sequential:
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(5, 5)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model

def fit_smaller_model(model):
    model_log = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                          validation_data=(x_test, y_test))
    return model_log

def run_smaller_model():
    model = smaller_model()
    model_log = fit_smaller_model(model)
    evaluate_model(model_log)

def evaluate_model(model_log):
    # score = model.evaluate(x_test, y_test, verbose=0

    print('Train loss: ', model_log.history['loss'][-1])
    print("Train accuracy: {}".format(model_log.history['accuracy'][-1]))
    print("Test loss: {}".format(model_log.history['val_loss'][-1]))
    print("Test accuracy: {}".format(model_log.history['val_accuracy'][-1]))

    plt.subplot(2, 1, 1)
    plt.plot(range(1, 11), model_log.history['accuracy'])
    plt.plot(range(1, 11), model_log.history['val_accuracy'])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="lower right")

    plt.subplot(2, 1, 1)
    plt.plot(range(1, 11), model_log.history["loss"])
    plt.plot(range(1, 11), model_log.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.tight_layout()
    plt.show()

    """
    If graph are similar for both train and test data, then no over-fitting is happening
    """
    return


if __name__ == "__main__":
    baseline_model()
    # run_basline_model()
    # run_smaller_model()


