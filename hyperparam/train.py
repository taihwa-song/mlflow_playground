import warnings

import math
import click
import mlflow

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

_random_state=37

@click.command(
    help="Train decision tree classifier on iris data"
)
@click.option("--max-depth", type=click.INT, default=None, help="The maximum depth of the tree.")
@click.option("--min-samples-leaf", type=click.INT, default=1, help="The minimum number of samples required to be at a leaf node.")
def run(max_depth, min_samples_leaf):
    warnings.filterwarnings("ignore")
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=_random_state)

    with mlflow.start_run():
        clf = DecisionTreeClassifier(max_depth=max_depth,
                                     min_samples_leaf=min_samples_leaf,
                                     random_state=_random_state)
        clf.fit(X_train, y_train)
        mlflow.sklearn.log_model(clf, "model")
        
        # Collect & log metrics
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        mlflow.log_metric("train_acc", train_acc)
        mlflow.log_metric("test_acc", test_acc)


if __name__ == "__main__":
    run()