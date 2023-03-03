# library doc string
'''
This module contains the customer churn prediction functionality
Author: Theresa Fruhwuerth
Date: 2.3.2023
'''
import logging
# import libraries
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, RocCurveDisplay

from sklearn.model_selection import GridSearchCV, train_test_split

import constants

sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'




def create_expected_directories():
    """
    Helper function to create directories necessary.
    output:
            None
    """
    for folder in constants.folders:
        if not os.path.exists(folder):
            os.mkdir(folder)


def import_data():
    '''
    returns dataframe for the csv found at datapath.
    output:
            dataframe: pandas dataframe
    '''
    return pd.read_csv(constants.datapath)


def _plot_correlation_matrix(dataframe):
    """
    Plot correlation matrix.
    input:
            dataframe: pandas dataframe
    output:
            None
    """
    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join(constants.imagepath, 'correlationmatrix.png'))
    plt.close()


def _plot_univariate_cat(dataframe, variablename='Income_Category'):
    """
    Plot univariate statistics for categorical variables.
    input:
            dataframe: pandas dataframe
            variablename: Name of the variable.
    output:
            None
    """
    dataframe[variablename].bar()
    plt.savefig(
        os.path.join(
            constants.imagepath,
            f'{variablename}_univariate_cat.png'))
    plt.close()


def _plot_univariate(dataframe, variablename='Churn'):
    """
    Plot univariate statistics for numeric variables.
    input:
            dataframe: pandas dataframe
            variablename: Name of the variable.
    output:
            None
    """
    dataframe[variablename].hist()
    plt.savefig(
        os.path.join(
            constants.imagepath,
            f'{variablename}_univariate.png'))
    plt.close()


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe.
    output:
            None
    '''
    logging.info("Performing EDA....")
    logging.info('##### Null values in data #####')
    logging.info(dataframe.isnull().sum())
    for variable in constants.plot_univar:
        _plot_univariate(dataframe, variablename=variable)
    for variable in constants.plot_univar_cat:
        _plot_univariate(dataframe, variablename=variable)
    _plot_correlation_matrix(dataframe)


def encoder_helper(dataframe, groupcol):
    '''
    Helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            groupcol: The column to group by categorical features
            response: string of response name [optional argument that could
            be used for naming variables or index y column]

    output:
            dataframe: pandas dataframe with new encoded column.
    '''

    groups = dataframe.groupby(groupcol).mean()[constants.dependent]
    dataframe[groupcol + '_' +
              constants.dependent] = dataframe[groupcol].apply(lambda x: groups[x])
    return dataframe


def preprocess_data(dataframe):
    """
    Preprocessing step to create the dependent variable before any other step.
    input:
            dataframe: pandas dataframe containing the data
    """
    dataframe[constants.dependent] = dataframe[constants.dependent_inputcol].apply(
        lambda val: 0 if val == constants.negative_class else 1)
    return dataframe


def perform_feature_engineering(dataframe):
    '''
    Perform the colmplete feature engineering.
    input:
              dataframe: pandas dataframe containing the data.
              could be used for naming variables or index y column]
              inputcol: string of original column name

    output:
              x_train: X training data
              x_test: X testing data
              ytrain: y training data
              ytest: y testing data
    '''
    logging.info("Preparing Data....")
    for groupcol in constants.cat_columns:
        dataframe = encoder_helper(dataframe, groupcol)
    x_data = dataframe[constants.keep_cols]
    ydata = dataframe[constants.dependent]
    x_train, x_test, ytrain, ytest = train_test_split(
        x_data, ydata, test_size=0.3, random_state=42)
    return x_train, ytrain, x_test, ytest


def prediction(model, x_test):
    '''
    Predicts with trained model on the test data.
    input:
            model: Trained model
            x_test: The Testdata
    output:
            ytest_pred: The predictions of the trained model.
    '''
    # scores
    print('random forest results')
    print('test results')
    ytest_pred = model.predict(x_test)
    return ytest_pred


def evaluation(model, x_test, ytest):
    '''
    Produces classification report for training and testing results and stores report as image
    in images folder.
        input:
                model: Trained model
                x_test: The Testdata
                ytest:  test response values
        output:
                 None
    '''
    logging.info("Evaluating Model....")
    ytest_pred = prediction(model, x_test)
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str(f'{constants.modelname} Train'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(ytest, ytest_pred)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(
        os.path.join(
            constants.imagepath,
            f'classificationreport_{constants.modelname}.png'))
    plt.close()
    RocCurveDisplay.from_predictions(ytest, ytest_pred)
    plt.savefig(
        os.path.join(
            constants.imagepath,
            f'roc_curve_{constants.modelname}.png'))
    plt.close()
    if constants.modeltraining:
        joblib.dump(
            model,
            os.path.join(
                constants.model_pth,
                f'{constants.modelname}_model.pkl'))


def feature_importance_plot(model, x_data):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
    output:
             None
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_data)
    shap.summary_plot(shap_values, x_data, plot_type="bar")
    plt.savefig(
        os.path.join(
            constants.imagepath,
            f'feature_importance_{constants.modelname}.png'))


def train_models(x_train, ytrain):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              ytrain: y training data
    output:
              trained_model: The model that was fitted.
    '''
    # grid search
    logging.info("Training Model....")
    if constants.modelname == 'rf':
        param_grid = [{'n_estimators': constants.n_estimators,
                       'min_samples_leaf': constants.minleafs}, ]
        model = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    elif constants.modelname == 'lrc':
        param_grid = [
            {'C': constants.log_c, 'penalty': constants.penalty},
        ]
        model = LogisticRegression(solver='lbfgs', max_iter=3000)
    else:
        raise NotImplementedError('Please implement this model for training.')
    crossvalidation = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5)
    crossvalidation.fit(x_train, ytrain)
    trainedmodel = crossvalidation.best_estimator_
    return trainedmodel


def main():
    '''
    Run the entire experiment workflow.
    '''
    dataframe = import_data()
    logging.info("Importing Data: SUCCESS")
    dataframe = preprocess_data(dataframe)
    logging.info("Preprocessing: SUCCESS")
    perform_eda(dataframe)
    logging.info("Finished EDA: SUCCESS")
    perform_eda(dataframe)

    x_train, ytrain, x_test, ytest = perform_feature_engineering(
        dataframe)
    logging.info("Preparing Data: SUCCESS")

    if constants.modeltraining:

        model = train_models(x_train, ytrain)
        logging.info("Training Model: SUCCESS")
    else:
        model = joblib.load(constants.bestmodel)
        logging.info("Loaded trained Model: SUCCESS")

    evaluation(model, x_test, ytest)
    feature_importance_plot(model, x_test)
    logging.info("Evaluating Model: SUCCESS")


if __name__ == "__main__":
    create_expected_directories()
    logging.basicConfig(
        filename='./logs/churn_library_main.log',
        level=logging.INFO,
        filemode='a',
        format='%(name)s - %(levelname)s - %(message)s')
    main()
