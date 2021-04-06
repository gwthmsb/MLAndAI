from keras import layers,  Sequential, utils
from keras.datasets import mnist

import matplotlib.pyplot as plt
from numpy import argmax

(X_train, y_train), (X_test, y_test) = mnist.load_data()


"""
Pre-processing:
1) Normalization
2) Categorizing output
"""

def pre_process(x_train, x_test):
    x_train = x_train.astype('float32')
    x_train /= 255
    x_test = x_test.astype('float32')
    x_test /= 255
    return x_train, x_test

def categorize_output(y_train, y_test):
    y_train = utils.np_utils.to_categorical(y_train, 10)
    y_test = utils.np_utils.to_categorical(y_test, 10)
    return y_train, y_test


"""
Define the model

1) We cannot send 2D array to NN. So, we flatten it to 1D vector
2) Define a dense layer of 128 neurons
3) Dropout
    * Dont change any inputs/output
    * 20% neurons are switched off in every iteration
    * This make sure that every neuron understand different characteristics
4) Output layer
5) Activation function - So that output is normalized

No of parameters: (# of neurons in previous layer + 1) * ( # neurons in next layer) 
"""

def initial_model():
    model = Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10))
    model.add(layers.Activation('softmax'))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    return model

def save_model(model):
    utils.plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True,
                     rankdir='TB', expand_nested=False, dpi=96)

def train_model(model, X_train, y_train, batch_size=32, epochs=10, verbose=0):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)


def evaluate(model, X_test, y_test, verbose=1):
    score = model.evaluate(X_test, y_test, verbose=verbose)
    return score

if __name__ == "__main__":
    model = initial_model()
    y_train, y_test = categorize_output(y_train, y_test)
    #save_model(initial_model())
    train_model(model, X_train, y_train, 32, 10, 0)
    score = evaluate(model, X_test, y_test, 1)
    print(score)

    test_index = 100
    plt.imshow(X_test[test_index])
    plt.show()
    x_test0 = X_test[test_index].reshape(-1, 28, 28)
    prediction = model.predict(x_test0)
    predicted_class = argmax(prediction)
    print(predicted_class)