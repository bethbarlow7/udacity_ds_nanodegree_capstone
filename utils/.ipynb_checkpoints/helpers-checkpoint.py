# Import libraries
import time

import numpy as np
import optuna
import pandas as pd
from sklearn import metrics
from sklearn.metrics import (
    classification_report,
    f1_score
)

from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

import matplotlib.pyplot as plt
import seaborn as sns


def fit_model_get_results(classifier, features, X_train, y_train, X_val, y_val):

    """
    Fits a model and gets results and execution times for train and test sets
    
    Inputs:
    classifier - defined model object
    features - list of features
    X_train - training feature set
    y_train - training labels
    X_val - validation feature set
    y_val - validation labels
    
    Outputs:
    model_results_dictionary - dictionary containing model object, f1-scores and execution times for training and predictions
    
    """
    start_time = time.time()

    trained_model = classifier.fit(X_train, y_train)
    training_execution_time = time.time() - start_time

    trained_model.feature_names = features

    # get predictions
    train_preds = classifier.predict(X_train)
    train_classification = classification_report(y_train, train_preds)

    train_f1 = f1_score(y_train, train_preds, average="weighted")

    start_time = time.time()
    val_preds = classifier.predict(X_val)
    prediction_execution_time = time.time() - start_time

    val_f1 = f1_score(y_val, val_preds, average="weighted")

    # get model and predictions dictionary
    model_results_dictionary = {
        "classifier": classifier,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "training_execution_time": np.round(training_execution_time, 0),
        "prediction_execution_time": np.round(prediction_execution_time, 0),
    }

    return model_results_dictionary


def plot_feature_importances(classifier, title, features):
    
    """
    Plots feature importances
    
    Inputs: 
    classifier - model object
    title - name of feature
    features - list of features
    
    Outputs:
    bar plot of features and feature importances
    
    """
    

    feature_importances_df = pd.DataFrame(
        {
            "Feature Importance": classifier.feature_importances_,
            "Feature Name": [v for v in features],
        }
    )

    plt.figure(figsize=(20, 13))

    sns.barplot(
        x="Feature Importance",
        y="Feature Name",
        data=feature_importances_df.loc[
            feature_importances_df["Feature Importance"] > 0
        ].sort_values(by="Feature Importance", ascending=False),
    )

    plt.title("Feature Importances - {}".format(title))
    plt.show()
