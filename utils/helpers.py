import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def fit_model_get_predictions(classifier, features):
    
    """
    Fits a model and gets predictions and execution times for train and test sets
    """
    start_time = time.time()
    
    trained_model = classifier.fit(X_train, y_train)
    training_execution_time = time.time() - start_time
                              
    trained_model.feature_names = features
    
    # get predictions
    train_preds = classifier.predict(X_train)
    train_classification = classification_report(y_train, train_preds)
    
    train_f1 = f1_score(y_train, train_preds, average = 'weighted')

    start_time = time.time()
    val_preds = classifier.predict(X_val)
    prediction_execution_time = time.time() - start_time
    
    val_f1 = f1_score(y_val, val_preds, average = 'weighted')
    
    # get model and predictions dictionary
    model_and_predictions_dictionary = {
        
        'classifier': classifier,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'training_execution_time': np.round(training_execution_time, 0),
        'prediction_execution_time': np.round(prediction_execution_time, 0)
        
    }
    
    return model_and_predictions_dictionary

def objective(trial):
    
    """
    Defines objective function for optuna to maximise.
    Takes a trial object as input and returns score
    """

    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, 50),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'eval_metric': 'auc',
    }
    
    clf = XGBClassifier(**params)
        
    start_time = time.time()
    clf.fit(X_train_res, y_train_res)
    training_execution_time = time.time() - start_time
    
    preds = np.rint(clf.predict(X_val))
    val_df['predictions'] = preds
        
    f1 = metrics.f1_score(y_val, preds, average = 'micro')
    
    return f1


def plot_feature_importances(model, title, features):

    feature_importances_df = pd.DataFrame(
        {
            "Feature Importance": model.feature_importances_,
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
