import keras
import numpy as np
import matplotlib.pyplot as plt

"""
mnist_fashion has 10 different fashion items

1) Training data:
2) Validation data:
3) Test data:  
"""
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
print("x_train shape:{}, y_train shape: {}".format(x_train.shape, y_train.shape))

# Further break the train data into validation and test data
(x_train, x_validation) = x_train[5000:], x_train[:5000]
(y_train, y_validation) = y_train[5000:], y_train[:5000]

# Reshape the input data to tensor(4D):
# 1) No of datapoints
# 2) Widths of the image
# 3) Height of the image
# 4) Channel to respresent( either color or grayscale)
w = h = 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_validation = x_validation.reshape(x_validation.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# Categorically labeling the outputs
y_train = keras.utils.to_categorical(y_train, 10)
y_validation = keras.utils.to_categorical(y_validation, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Defining LeNet model
def build_lenet_model():
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(filters=6, kernel_size=5, activation="sigmoid",
                                  padding='same', input_shape=(28, 28, 1)))
    # Input shape: (28, 28, 6), output shape:(14, 14, 6)
    model.add(keras.layers.AvgPool2D(pool_size=2, strides=2))
    # count = 16*(5*5*6 +1) = 2416. 6- is from previous layer(Depth)
    # Input shape: (14, 14, 6) Output shape: (n-k+1, n-k+1, filters): (10, 10, 16)
    model.add(keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'))
    # Input shape: (10, 10, 16) Output shape: (5, 5, 16)
    model.add(keras.layers.AvgPool2D(pool_size=2, strides=2))
    # Input shape: (5, 5, 16) Output shape: 5*5*16 = 400
    model.add(keras.layers.Flatten())
    # Input shape: 400,
    # Neurons: 120
    # Parameters: (400+1)*120 = 48120
    model.add(keras.layers.Dense(120, activation='sigmoid'))
    # Parameters: (120+1)*84 = 10164
    model.add(keras.layers.Dense(84, activation='sigmoid'))
    # Parameters: (84+1)*10 = 850
    model.add(keras.layers.Dense(10, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model


def train_basic_model():
    model = build_lenet_model()
    # Model checkpoint: save the weights at that level
    checkpointer = keras.callbacks.ModelCheckpoint(filepath="model.weights.best", verbose=1,
                                                   save_best_only=True)
    model_log = model.fit(x_train, y_train, batch_size=64, validation_data=(x_validation, y_validation),
                          epochs=2, callbacks=[checkpointer])


def test_basic_model(model):
    model.load_weights("model.weights.best")
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test accuracy: {}".format(score[1]))


def build_model_variant()->keras.Sequential:
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
                                  input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model


if __name__=="__main__":
    # test_basic_model(build_lenet_model())
    build_model_variant()