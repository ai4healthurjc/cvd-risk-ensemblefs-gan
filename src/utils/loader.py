import numpy as np
import pandas as pd
import utils.consts as consts
from pathlib import Path
import keras
from scikeras.wrappers import KerasClassifier, KerasRegressor
import pickle
import math
import joblib
from utils.metrics import compute_mrae, compute_mae
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


list_diabetes_categorical_cols = ['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia',
                                  'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing',
                                  'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity', 'class']
list_diabetes_numerical_cols = ['Age']


def save_model_estimator(best_estimator, fes, type_over, estimator_name, seed_value):

    path_estimator_model = str(Path.joinpath(consts.PATH_PROJECT_MODELS,
                                             "model_{}_{}_{}_{}".format(fes, type_over, estimator_name, seed_value))
                               )

    if estimator_name == 'mlp':
        logger.info('Saving mlp model')
        best_estimator.model_.save('{}.h5'.format(path_estimator_model))
    else:
        joblib.dump(best_estimator, '{}.joblib'.format(path_estimator_model))


def load_model_estimator(fes, type_over, estimator, seed_value, x_train, y_train):
    path_estimator_model = Path.joinpath(consts.PATH_PROJECT_MODELS,
                                         "model_{}_{}_{}_{}".format(fes, type_over, estimator, seed_value)
                                         )

    if estimator == 'mlp':
        str_path_model = '{}.{}'.format(str(path_estimator_model), 'h5')
        new_reg_model = keras.models.load_model(str_path_model, custom_objects={"compute_mrae": compute_mrae})
        reg_new = KerasRegressor(new_reg_model)
        reg_new.initialize(x_train, y_train)
        return reg_new
    else:
        # path_estimator_model = Path.joinpath(consts.PATH_PROJECT_MODELS,
        #                                      "model_{}_{}_{}_{}.joblib".format(fes, type_over, estimator, seed_value)
        #                                      )
        str_path_model = '{}.{}'.format(str(path_estimator_model), 'joblib')
        loaded_model = joblib.load(str_path_model)
        return loaded_model


def get_categorical_numerical_names(df_data: pd.DataFrame, bbdd_name: str) -> (list, list):

    df_info = identify_type_features(df_data)
    list_numerical_vars = list(df_info[df_info['type'] == consts.TYPE_FEATURE_CONTINUOUS].index)
    list_categorical_vars = list(df_info[df_info['type'] == consts.TYPE_FEATURE_DISCRETE].index)

    return list_categorical_vars, list_numerical_vars


def identify_type_features(df, discrete_threshold=10, debug=False):
    """
    Categorize every feature/column of df as discrete or continuous according to whether or not the unique responses
    are numeric and, if they are numeric, whether or not there are fewer unique renponses than discrete_threshold.
    Return a dataframe with a row for each feature and columns for the type, the count of unique responses, and the
    count of string, number or null/nan responses.
    """
    counts = []
    string_counts = []
    float_counts = []
    null_counts = []
    types = []
    for col in df.columns:
        responses = df[col].unique()
        counts.append(len(responses))
        string_count, float_count, null_count = 0, 0, 0
        for value in responses:
            try:
                val = float(value)
                if not math.isnan(val):
                    float_count += 1
                else:
                    null_count += 1
            except:
                try:
                    val = str(value)
                    string_count += 1
                except:
                    print('Error: Unexpected value', value, 'for feature', col)

        string_counts.append(string_count)
        float_counts.append(float_count)
        null_counts.append(null_count)
        types.append('d' if len(responses) < discrete_threshold or string_count > 0 else 'c')

    feature_info = pd.DataFrame({'count': counts,
                                 'string_count': string_counts,
                                 'float_count': float_counts,
                                 'null_count': null_counts,
                                 'type': types}, index=df.columns)
    if debug:
        print(f'Counted {sum(feature_info["type"] == "d")} discrete features and {sum(feature_info["type"] == "c")} continuous features')

    return feature_info


def load_train_test_partitions(filename_pattern, seed_value):

    df_x_train_all_pre = pd.read_csv(
        str(Path.joinpath(consts.PATH_PROJECT_DATA_PARTITIONS,
                          'df_x_train_{}.csv'.format(filename_pattern))),
    )
    df_x_test_all_pre = pd.read_csv(
        str(Path.joinpath(consts.PATH_PROJECT_DATA_PARTITIONS,
                          'df_x_train_{}.csv'.format(filename_pattern))),
    )

    df_x_train_all_raw = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PARTITIONS, 'df_x_train_raw_{}.csv'.format(seed_value))))
    df_x_test_all_raw = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PARTITIONS, 'df_x_test_raw_{}.csv'.format(seed_value))))

    list_low = df_x_test_all_pre.index[(df_x_test_all_pre['label'] >= 0.0) & (df_x_test_all_pre['label'] < 0.1)].to_list()
    list_moderate = df_x_test_all_pre.index[(df_x_test_all_pre['label'] >= 0.1) & (df_x_test_all_pre['label'] < 0.2)].to_list()
    list_high = df_x_test_all_pre.index[df_x_test_all_pre['label'] >= 0.2].to_list()

    dict_list_patients = {
        'cvd_low': list_low,
        'cvd_moderate': list_moderate,
        'cvd_high': list_high
    }

    y_train = df_x_train_all_pre['label'].values
    df_x_train_raw = df_x_train_all_raw.drop(columns=['label'])
    df_x_train_pre = df_x_train_all_pre.drop(columns=['label'])

    y_test = df_x_test_all_pre['label'].values
    df_x_test_raw = df_x_test_all_raw.drop(columns=['label'])
    df_x_test_pre = df_x_test_all_pre.drop(columns=['label'])

    return df_x_train_pre, y_train, df_x_test_pre, y_test, df_x_train_raw, df_x_test_raw, dict_list_patients


def load_raw_dataset(bbdd_name: str) -> pd.DataFrame:
    df_data = None
    if bbdd_name in consts.LISTS_BBDD_CLINICAL:
        df_data = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_RAW, bbdd_name, 'bbdd_{}.csv'.format(bbdd_name))))
    else:
        ValueError('Dataset not found!')

    return df_data


def load_preprocessed_dataset(bbdd_name: str, show_info=False) -> (np.array, np.array, np.array, list, list):

    df_data = None

    if bbdd_name in consts.LISTS_BBDD_CLINICAL or bbdd_name in consts.LISTS_BBDD_GENERAL:
        df_data = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED, 'bbdd_{}.csv'.format(bbdd_name))))
    else:
        ValueError('Dataset not found!')

    y_label = df_data['label'].values
    df_features = df_data.drop(['label'], axis=1)
    v_column_names = df_features.columns.values
    m_features = df_features.values

    list_vars_categorical, list_vars_numerical = get_categorical_numerical_names(df_features, bbdd_name)

    if show_info:
        logger.info('n_samples: {}, n_features: {}'.format(df_features.shape[0], df_features.shape[1]))
        logger.info('Classes: {}, # samples: {}'.format(np.unique(y_label, return_counts=True)[0],
                                                        np.unique(y_label, return_counts=True)[1]))
        logger.info('List of numerical features {}, #{}'.format(list_vars_numerical, len(list_vars_numerical)))
        logger.info('List of categorical features {}, #{}'.format(list_vars_categorical, len(list_vars_categorical)))

    return m_features, y_label, v_column_names, list_vars_categorical, list_vars_numerical
    # return m_features, y_label, v_column_names
