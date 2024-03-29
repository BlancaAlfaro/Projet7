import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit


def get_initial_splits(df,test_size=0.3):
    """Split initial dataframe into parts used for training, test and verification of score on Kaggle.

    Parameters
    ----------
    df : pd.DataFrame
        initial Dataframe containing features and target
    test_size : float, optional
        proportion to keep for testing, by default 0.3

    Returns
    -------
    (pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.Series,pd.Series)
        Dataframes for kaggle verification, X_train, X_test, y_train and y_test.
    """
   #Split into working part and part that will be submitted to Kaggle to check score
    wrk_df = df[df['TARGET'].notnull()].reset_index(drop=True)
    kaggle_df = df[df['TARGET'].isnull()].drop(columns='TARGET')
    X=wrk_df.copy()
    y=X.pop('TARGET')
    # From X we will need a test split for final validation and a train split which will be later divided into several folds of training and validations parts.
    # Since classes are inbalanced, train/test split will be done with StratifiedShuffleSplit.
    splitter=StratifiedShuffleSplit(n_splits=1,test_size=test_size,random_state=33)
    splits=splitter.split(X,y)
    for _, (train_index, test_index) in enumerate(splits):
        X_train=X.loc[train_index]
        X_test=X.loc[test_index]
        y_train=y.loc[train_index]
        y_test=y.loc[test_index]
    return kaggle_df,X_train,X_test,y_train,y_test

def get_grid_cv_scores(model,params,custom_score,X_train,y_train):
    """ Perform grid search cv with a given model and parameters using a custom score and return the best model as well as the scores obtaines for each parameter set.

    Parameters
    ----------
    model : sklearn model
        A model compatible with sklearn Base Estimator
    params : dict
        Dictionnary of parameters to pass to GridSearchCV
    custom_score : score
        scorer made with make_scorer from sklearn.metrics to use to select best model
    X_train : pd.DataFrame, optional
        Trainning dataset's features, by default X_train
    y_train : pd.Series, optional
        Target for training, by default y_train

    Returns
    -------
    tuple : (sklearn model , pd.DataFrame)
        A tuple containing the best model as defined by the custom_score and the scoring results of all parameter sets.
    """

    # Define all the scoring metrics to compute
    scoring = {"AUC": "roc_auc", "Accuracy": "accuracy", "F1-score": "f1", "Custom_scorer" : custom_score}

    pipeline = Pipeline([
            ('sampling', SMOTE()),
            ('classification', model)
        ])

    grid=GridSearchCV(pipeline,
                    param_grid=params,
                    cv=5,
                    refit="Custom_scorer",
                    scoring=scoring,
                    return_train_score=True)

    grid.fit(X_train, y_train)

    #Extract the scoring results to keep
    results = pd.DataFrame(grid.cv_results_)[['mean_fit_time','params',
                                              'mean_train_AUC','mean_train_Accuracy','mean_train_F1-score','mean_train_Custom_scorer',
                                              'mean_test_AUC','mean_test_Accuracy','mean_test_F1-score','mean_test_Custom_scorer']]

    return grid.best_estimator_, results



def plot_metric_results(results):
    """Plot mean scores obtained for the following metrics from the GridSearchCV results :
    fit time, AUC, Accuracy, F1-score, custom score

    Parameters
    ----------
    results : pd.DataFrame
        results of GridSearchCV containing for each parameter set (rows) the wanted scores to plot for both training and validation splits.
    """
    _,axs=plt.subplots(3,2,figsize=(18,20))
    idx=["setting_"+str(i) for i in range(results.shape[0])]

    sns.lineplot(x=idx,y=results['mean_fit_time'],ax=axs[0][0],label='mean_fit_time')
    axs[0][0].tick_params('x', labelrotation=90)

    sns.lineplot(x=idx,y=results['mean_train_AUC'],ax=axs[0][1],label='mean_train_AUC')
    sns.lineplot(x=idx,y=results['mean_test_AUC'],ax=axs[0][1],label='mean_test_AUC')
    axs[0][1].tick_params('x', labelrotation=90)

    sns.lineplot(x=idx,y=results['mean_train_Accuracy'],ax=axs[1][0],label='mean_train_Accuracy')
    sns.lineplot(x=idx,y=results['mean_test_Accuracy'],ax=axs[1][0],label='mean_test_Accuracy')
    axs[1][0].tick_params('x', labelrotation=90)

    sns.lineplot(x=idx,y=results['mean_train_F1-score'],ax=axs[1][1],label='mean_train_F1-score')
    sns.lineplot(x=idx,y=results['mean_test_F1-score'],ax=axs[1][1],label='mean_test_F1-score')
    axs[1][1].tick_params('x', labelrotation=90)

    sns.lineplot(x=idx,y=results['mean_train_Custom_scorer'],ax=axs[2][0],label='mean_train_Custom_scorer')
    sns.lineplot(x=idx,y=results['mean_test_Custom_scorer'],ax=axs[2][0],label='mean_test_Custom_scorer')
    axs[2][0].tick_params('x', labelrotation=90)

    axs[2][1].axis('off')
    plt.show()