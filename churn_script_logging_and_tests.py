'''
Tests for the functionality of the churn library.
Author: Theresa Fruhwuerth
Date: 2.3.2023
'''

import logging
import os

import pytest
from sklearn.utils.validation import check_is_fitted
from copy import deepcopy
import churn_library
import constants
import numpy as np



def setup():
    """
    Set up directory for image testing.
    :param directory: The directory to remove and create.
    :return: None
    """
    for folder in constants.folders:
        if not os.path.exists(folder):
            os.system(f"rm -rf {folder}")
            os.mkdir(folder)


@pytest.fixture(name='dataframe_input', scope='function')
def fixutre_dataframe():
    """Read input dataframe for tests."""
    return deepcopy(churn_library.import_data())


@pytest.fixture(name='datasets', scope='function')
def fixture_datasets(dataframe_input):
    """
    Data fixture to load data only once during test run.
    :return:
        Data in various forms incl. engineered features
    """
    preprocessed_data = churn_library.preprocess_data(dataframe_input)
    x_train, ytrain, x_test, y_test = churn_library.perform_feature_engineering(
        preprocessed_data)

    return x_train, ytrain, x_test, y_test, preprocessed_data


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe_in = churn_library.import_data()
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe_in.shape[0] > 0
        assert dataframe_in.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_preprocess_data(dataframe_input):
    '''
    Test preprocessing function
    '''
    preprocessed_data = churn_library.preprocess_data(dataframe_input)
    try:
        assert constants.dependent in preprocessed_data.columns
    except AssertionError:
        logging.error('Target column not in columns.')


def test_eda(datasets):
    '''
    test perform eda function
    '''
    _, _, _, _, preprocessed_data = datasets
    churn_library.perform_eda(preprocessed_data)
    try:
        assert len(os.listdir(constants.imagepath)) == 3
    except AssertionError:
        logging.error('Not all image files where saved, '
                      'saved images: %s', str(os.listdir(constants.imagepath)))


def test_encoder_helper(datasets):
    '''
    test encoder helper
    '''
    _, _, _, _, preprocessed_data = datasets
    churn_library.perform_eda(preprocessed_data)
    original_columns = preprocessed_data.columns
    try:
        assert constants.dependent in preprocessed_data.columns
    except AssertionError:
        logging.error('Dependent variable not in data.')
    for col in constants.cat_columns:
        preprocessed_data = churn_library.encoder_helper(
            preprocessed_data, col)
    try:
        assert len(original_columns) + \
               len(constants.cat_columns) == len(preprocessed_data.columns)
    except AssertionError:
        logging.error('Number of columns after encoding not as expected.')
    diffcols = set(original_columns).difference(preprocessed_data.columns)
    try:
        assert [x + 'Churn' for x in constants.cat_columns] == list(diffcols)
    except AssertionError:
        logging.error('Column content not as expected.')


def test_perform_feature_engineering(datasets):
    '''
    test perform_feature_engineering
    '''
    _, _, _, _, preprocessed_data = datasets
    x_train, ytrain, x_test, ytest = churn_library.perform_feature_engineering(
        preprocessed_data)
    try:
        assert x_test.shape[0] == ytest.shape[0]
        assert x_train.shape[0] == ytrain.shape[0]
    except AssertionError:
        logging.error('Shape of test or train not similar.')
    try:
        assert set([x + 'Churn' for x in constants.cat_columns]
                   ).intersection(x_test.columns) == len(constants.cat_columns)
        assert set([x + 'Churn' for x in constants.cat_columns]
                   ).intersection(x_train.columns) == len(constants.cat_columns)
    except AssertionError:
        logging.error('Expected categorical columns missing.')


def test_train_evaluate_models(datasets):
    '''
    test train_models
    '''
    x_train, ytrain, x_test, ytest, _ = datasets
    model = churn_library.train_models(x_train, ytrain)
    try:
        assert check_is_fitted(model) is True
    except AssertionError:
        logging.error('Model not fitted.')
    y_pred = churn_library.prediction(model, x_test)
    try:
        assert sorted(np.unique(y_pred).tolist()) == [0, 1]
    except AssertionError:
        logging.error('Predictions take  on unexpected values.')
    try:
        modelfiles = os.listdir('./models/')
        assert len([x for x in modelfiles if '.pkl' in x]) >= 1
    except AssertionError:
        logging.error('Model not saved.')
    churn_library.evaluation(model, x_test, ytest)
    resultplots = os.listdir(constants.imagepath)
    try:
        assert len([x for x in resultplots if 'classificationreport' in x]) == 1
        assert len([x for x in resultplots if 'roc' in x]) == 1
    except AssertionError:
        logging.error('Missing result plot.')


if __name__ == "__main__":
    setup()
    logging.basicConfig(
        filename='./logs/churn_librarytests.log',
        level=logging.INFO,
        filemode='a',
        format='%(name)s - %(levelname)s - %(message)s')
