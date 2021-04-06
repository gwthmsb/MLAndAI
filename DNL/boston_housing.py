from keras.datasets import boston_housing

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pandas

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
print(X_train[0], y_train[0])
# There are 13 features


def run_model(pipeline, X_train, y_train, model_name="Baseline"):
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X_train, y_train, cv=kfold)
    print(model_name+": %.2f (%.2f) MSE" % (results.mean(), results.std()))


def baseline_model() -> Sequential:
    model = Sequential()
    # Dense: 13- No of neurons
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.summary()
    return model


def evaluate_baseline():
    #Evaluating baseline
    estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10)
    results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    print("Baseline: %.2f (%.2f) MSE"%(results.mean(), results.std()))
    # Baseline: -24.26 (11.43) MSE


""" 
3. Standardizing: Tuning further
Considered part of pre-processing
Standardizing: Process of normalizing the different features to uniform scale

Calculating overall avg

"""
def standardize_baseline():
    estimators = []
    estimators.append(("strandardize", StandardScaler()))
    estimators.append(("mlp", KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
    run_model(Pipeline(estimators), X_train, y_train, "Standardized: ")
    # Standardized: -22.14 (10.21) MSE


"""
4. Tuning with deeper models
Adding extra neuron layers

"""
def larger_model():
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model


def tune_with_bigger_model():
    estimators = []
    estimators.append(("standardize", StandardScaler()))
    estimators.append(("mlp", KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
    run_model(Pipeline(estimators), X_train, y_train, "Larger model")
    # Larger model: -17.53 (9.21) MSE


"""
Tuning with wider model

Process of varying hyper-parameters of the layers
One approach is to increase number of neurons in layer

"""

def wider_model():
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model

def tune_with_wider_model():
    estimators = []
    estimators.append(("standardize", StandardScaler()))
    estimators.append(("mlp", KerasRegressor(build_fn=wider_model, epochs=50, batch_size=5, verbose=0)))
    run_model(Pipeline(estimators), X_train, y_train, "Tune with wider model")


if __name__ == "__main__":
    # evaluate_baseline()
    # standardize_baseline()
    # tune_with_bigger_model()
    tune_with_wider_model()
