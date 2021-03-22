from numpy import mean
from numpy import std
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

"""
    LinearDiscriminantAnalysis is classifier not an reducer. 
"""


def check_model(X, y, model):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print(scores)
    print("Mean accuracy: %.3f, SD: %.3f"%(mean(scores), std(scores)))


def simple_prediction(X, y, model):
    model.fit(X, y)
    row = [0,0,-2.23268854,-1.82114386,1.75466361,0.1243966,1.03397657,2.35822076,4.44235,0.56768485]
    # make a prediction
    yhat = model.predict([row])
    # summarize prediction
    print('Predicted Class: %d' % yhat)


def hyper_parameters(X, y, model, solver):
    """
        Hyper parameters should be configured for LDA.
        Importanta HP is solver, by default it is "svd"
    """
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid = {"solver": [solver,]}
    search_model = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
    results = search_model.fit(X, y)
    print("Solver: {}".format(results.best_params_))
    print("Accuracy: %.8f"%results.best_score_)
    return results


def iris_data():
    # define dataset
    #X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
    X, y = load_iris(True)
    print(X, y)
    # define model
    model = LinearDiscriminantAnalysis()
    print(model.get_params(True))
    #simple_prediction(X, y, model)
    solvers = ["svd", "lsqr", "eigen"]
    row = [0.1, 3.5, 4.2, 100]
    for solver in solvers:
        result = hyper_parameters(X, y, model, solver)
        pr_class = result.predict([[10, 25, 30, 40]])
        print(pr_class)


def iris_data_reduction():
    X, y = load_iris(True)
    model = LinearDiscriminantAnalysis(n_components=2)
    model.fit(X, y)
    X_trans = model.transform(X)
    print(X_trans)
    model.fit(X_trans, y)
    print(model.predict([[0, 0]]))
    print(model.predict([[100, 100]]))
    print(model.predict([[-100, -100]]))


def prediction():
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
    model = LinearDiscriminantAnalysis()
    simple_prediction(X, y, model)


def LDA_as_reduction():
    X, y = make_classification(n_samples=10, n_features=6, n_informative=6, n_redundant=0, random_state=2, n_classes=3)
    model = LinearDiscriminantAnalysis(n_components=2)
    print(X, y)
    print(model.get_params(True))
    model.fit(X, y)
    print(model.predict([[2, 4, 5, -1, 0, 4]]))
    X_trans = model.transform(X)
    print(X_trans)
    model.fit(X_trans, y)
    print(model.predict([[2, 4]]))
    print(model.predict([[-3, 5]]))


def LDA_as_reduction_using_pipeline():
    steps = [("lda", LinearDiscriminantAnalysis()), ("m", GaussianNB())]


if __name__ == "__main__":
    #prediction()
    #iris_data()
    #iris_data_reduction()
    LDA_as_reduction()