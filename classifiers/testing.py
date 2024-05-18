from enum import Enum
from typing import Callable, Iterable

import pandas as pd
from sklearn.metrics import classification_report

import classifiers.ml_dataset_preparation as prep


class TestType(Enum):
    ALL = 0
    FIVE_VS_ALL = 1
    THREE_LEVELS = 2

    def __str__(self) -> str:
        if self == TestType.ALL:
            return 'All classes'
        if self == TestType.FIVE_VS_ALL:
            return '5 Vs All'
        return 'Three Levels'


class DatasetFeatures(Enum):
    WITH_TEXT_TONE_AS_PERCENTAGES = 0
    WITH_TEXT_TONE_AS_COUNTS = 1
    WITHOUT_TEXT_TONE = 2

    def __str__(self) -> str:
        if self == DatasetFeatures.WITHOUT_TEXT_TONE:
            return 'Without DA features'
        if self == DatasetFeatures.WITH_TEXT_TONE_AS_COUNTS:
            return 'With DA features (counts)'
        return 'With DA features (%)'


class TestInputs:

    def __init__(self,
                 x_train,
                 x_test,
                 y_train,
                 y_test,
                 test_type: TestType,
                 dataset_features: DatasetFeatures) -> None:
        self.dataset_features = dataset_features
        self.test_type = test_type
        self.y_test = y_test
        self.y_train = y_train
        self.x_test = x_test
        self.x_train = x_train


def train_and_test(X,
                   fit_predict: Callable[[TestInputs], Iterable],
                   train_size,
                   y,
                   test_type: TestType,
                   dataset_features: DatasetFeatures):
    # TODO we should pass the test type to the fit_predict and pass an object
    x_train, x_test, y_train, y_test = prep.split(X, y, train_size=train_size)
    # print('Y train labels: ')
    # print(y_train.value_counts())
    # print('Y test labels: ')
    # print(y_test.value_counts())

    predicted = fit_predict(TestInputs(x_train, x_test, y_train, y_test, test_type, dataset_features))
    print(f'{test_type} - {dataset_features}')
    print(classification_report(y_test, predicted))


def cycle_test(test_name: str,
               fit_predict: object,
               ml_dataset_path: str = './temp_data/scored_issues_snapshots_w2v_cls.csv',
               ml_dataset_idx: str = 'idx',
               train_size: float = 0.7,
               test_type: TestType = TestType.FIVE_VS_ALL,
               dataset_types=[DatasetFeatures.WITHOUT_TEXT_TONE,
                              DatasetFeatures.WITH_TEXT_TONE_AS_PERCENTAGES,
                              DatasetFeatures.WITH_TEXT_TONE_AS_COUNTS],
               pre_processor: object = None,
               drop_categories: bool = True,
               add_dummies: bool = True):
    if pre_processor is None:
        def noup(x, y):
            return x, y

        pre_processor = noup
    print('=' * 50)
    print(f'start test {test_name} to test {test_type}')
    issues_df = pd.read_csv(ml_dataset_path, index_col=[ml_dataset_idx])
    print(F'Total records in dataset %i' % len(issues_df))

    if type(dataset_types) == DatasetFeatures:
        dataset_types = [dataset_types]

    class_to_predict = class_to_predict_by_test_type(test_type)
    X, y = prep.build_dataset(issues_df.copy(), class_to_predict=class_to_predict, utterances_as_percentage=True,
                              drop_categories=drop_categories, add_dummies=add_dummies)
    X, y = pre_processor(X, y)
    if DatasetFeatures.WITHOUT_TEXT_TONE in dataset_types:
        test_without_text_tone_features(X, fit_predict, test_type, train_size, y)

    if DatasetFeatures.WITH_TEXT_TONE_AS_PERCENTAGES in dataset_types:
        test_with_text_tone_as_percentages(X, fit_predict, test_type, train_size, y)

    if DatasetFeatures.WITH_TEXT_TONE_AS_COUNTS in dataset_types:
        test_with_text_tone_as_counts(class_to_predict, fit_predict, issues_df, test_type, train_size, pre_processor,
                                      drop_categories, add_dummies)
    print('=' * 50)


def class_to_predict_by_test_type(test_type):
    class_to_predict = None
    if test_type == TestType.FIVE_VS_ALL:
        class_to_predict = 5
    if test_type == TestType.THREE_LEVELS:
        class_to_predict = {5: 2,
                            4: 1,
                            3: 1,
                            2: 0,
                            1: 0}
    return class_to_predict


def test_with_text_tone_as_counts(class_to_predict, fit_predict, issues_df, test_type, train_size, pre_processor,
                                  drop_categories, add_dummies):
    X, y = prep.build_dataset(issues_df.copy(), class_to_predict=class_to_predict, utterances_as_percentage=False,
                              drop_categories=drop_categories, add_dummies=add_dummies)
    X, y = pre_processor(X, y)
    train_and_test(X,
                   fit_predict,
                   train_size,
                   y,
                   test_type,
                   DatasetFeatures.WITH_TEXT_TONE_AS_COUNTS)


def test_with_text_tone_as_percentages(X, fit_predict, test_type, train_size, y):
    train_and_test(X,
                   fit_predict,
                   train_size,
                   y,
                   test_type,
                   DatasetFeatures.WITH_TEXT_TONE_AS_PERCENTAGES)


def test_without_text_tone_features(X, fit_predict, test_type, train_size, y):
    train_and_test(prep.drop_utterances_classification_features(X),
                   fit_predict,
                   train_size,
                   prep.drop_utterances_classification_features(y),
                   test_type,
                   DatasetFeatures.WITHOUT_TEXT_TONE)


def print_heading(heading):
    print("*" * 10, end='')
    print(heading, end='')
    print("*" * 10)
