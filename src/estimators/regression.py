import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from pathlib import Path
import datetime
import tensorflow as tf
import random as python_random
from sklearn.preprocessing import StandardScaler
import logging
import coloredlogs

import utils.consts as consts
from fs.filter import Relief, Surf, PermutationImportance, MRMR
from estimators.mlp import MLPKerasRegressor
from utils.metrics import compute_mrae
from utils.plotter import plot_fs_score, plot_performance_evolution_k_features


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

TOT = 1e-4
MAX_ITERS = 20000
TEST_SIZE_PROPORTION = 0.2
VALIDATION_SIZE_PROPORTION = 0.2


def scale_features(x_train, y_train,
                   x_test, y_test,
                   v_col_names_ohe,
                   list_vars_numerical,
                   list_vars_categorical,
                   seed_value,
                   save_data=False
                   ):

    scaler_num = StandardScaler()
    df_x_train = pd.DataFrame(x_train, columns=v_col_names_ohe)
    df_x_test = pd.DataFrame(x_test, columns=v_col_names_ohe)
    df_x_train_raw = df_x_train.copy()
    df_x_test_raw = df_x_test.copy()
    scaler_num.fit(df_x_train[list_vars_numerical].values)
    df_x_train[list_vars_numerical] = scaler_num.transform(df_x_train[list_vars_numerical].values)
    df_x_test[list_vars_numerical] = scaler_num.transform(df_x_test[list_vars_numerical].values)

    if save_data:
        df_x_train_raw['label'] = y_train
        df_x_test_raw['label'] = y_test

        df_x_train_pre = df_x_train.copy()
        df_x_test_pre = df_x_test.copy()
        df_x_train_pre['label'] = y_train
        df_x_test_pre['label'] = y_test

        df_x_train_raw.to_csv(
            str(Path.joinpath(consts.PATH_PROJECT_DATA_PARTITIONS, 'df_x_train_raw_{}.csv'.format(seed_value))),
            index=False
        )
        df_x_test_raw.to_csv(
            str(Path.joinpath(consts.PATH_PROJECT_DATA_PARTITIONS, 'df_x_test_raw_{}.csv'.format(seed_value))),
            index=False
        )
        df_x_train_pre.to_csv(
            str(Path.joinpath(consts.PATH_PROJECT_DATA_PARTITIONS, 'df_x_train_{}.csv'.format(seed_value))),
            index=False
        )
        df_x_test_pre.to_csv(
            str(Path.joinpath(consts.PATH_PROJECT_DATA_PARTITIONS, 'df_x_test_{}.csv'.format(seed_value))),
            index=False
        )

    return df_x_train.values, df_x_test.values, df_x_train_raw, df_x_test_raw


def compute_evolution_k_parameters(df_features, y_label, v_col_names,
                                   estimator_name, fs_method_name, type_over, join_synthetic,
                                   list_vars_numerical, list_vars_categorical):

    df_metrics_total = pd.DataFrame(columns=['estimator', 'fs_method', 'k_features', 'metric', 'mean', 'std'])

    for fs_method_name in [fs_method_name]:
        df_metrics_by_fs = train_fs_with_different_k_features(df_features, y_label, v_col_names,
                                                              fs_method_name, estimator_name,
                                                              list_vars_numerical, list_vars_categorical,
                                                              join_synthetic, type_over)

        df_metrics_total = df_metrics_total.append(df_metrics_by_fs, ignore_index=True)
        print(df_metrics_total)

    plot_performance_evolution_k_features(df_metrics_total, 'mae', estimator_name, fs_method_name, type_over,
                                          flag_save_figure=True)
    plot_performance_evolution_k_features(df_metrics_total, 'mrae', estimator_name, fs_method_name, type_over,
                                          flag_save_figure=True)


def encode_albuminuria(x_train, y_train, x_test, y_test, v_col_names):

    df_x_train = pd.DataFrame(x_train, columns=v_col_names)
    df_x_test = pd.DataFrame(x_test, columns=v_col_names)

    df_train_album_binary = pd.get_dummies(df_x_train.album, prefix='album', dtype=int)
    df_test_album_binary = pd.get_dummies(df_x_test.album, prefix='album', dtype=int)

    df_x_train['album_0'] = df_train_album_binary['album_0.0']
    df_x_train['album_1'] = df_train_album_binary['album_1.0']
    df_x_train['album_2'] = df_train_album_binary['album_2.0']

    df_x_test['album_0'] = df_test_album_binary['album_0.0']
    df_x_test['album_1'] = df_test_album_binary['album_1.0']
    df_x_test['album_2'] = df_test_album_binary['album_2.0']

    df_x_train = df_x_train.drop(columns=['album'])
    df_x_test = df_x_test.drop(columns=['album'])

    v_col_names = df_x_train.columns.values

    return df_x_train.values, y_train, df_x_test.values, y_test, v_col_names


def get_hyperparams(estimator_name: str, num_features: int = 5, seed_value: int = 4242):

    dict_param_grid_regressor = {}
    regressor = None

    logger.info('Selected estimator: {}'.format(estimator_name))

    if estimator_name == 'svr':

        dict_param_grid_regressor = {
            'C': np.linspace(0.01, 10),
            'gamma': ['auto', 'scale']
        }

        regressor = SVR(max_iter=MAX_ITERS)

    elif estimator_name == 'ridge':
        dict_param_grid_regressor = {
            'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0],
            "fit_intercept": [True, False],
            "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }

        regressor = Ridge()

    elif estimator_name == 'dt':

        dict_param_grid_regressor = {
            'max_depth': np.arange(1, 15, 1),
            'min_samples_split': np.arange(2, 20, 2)
        }

        regressor = DecisionTreeRegressor(random_state=seed_value)

    elif estimator_name == 'rf':

        dict_param_grid_regressor = {
            'n_estimators': [20, 30, 40, 50],
            'min_samples_split': [2, 4, 8, 10]
        }

        regressor = RandomForestRegressor(random_state=seed_value)

    elif estimator_name == 'elasticnet':

        dict_param_grid_regressor = {
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0],
            "l1_ratio": np.arange(0, 1, 0.01)
        }

        regressor = ElasticNet(max_iter=MAX_ITERS, random_state=seed_value)

    elif estimator_name == 'knn':

        dict_param_grid_regressor = {
            'n_neighbors': np.arange(1, 12, 1),
        }

        regressor = KNeighborsRegressor()

    elif estimator_name == 'mlp':

        np.random.seed(seed_value)
        python_random.seed(seed_value)
        tf.random.set_seed(seed_value)

        # early_stopping = EarlyStopping(monitor='val_mean_absolute_error', patience=5, verbose=0, mode='min')

        fit_kwargs = {
            "batch_size": 32,
            # "batch_size": 64,
            "validation_split": 0.15,
            "verbose": 0,
            "epochs": 500,
            "n_features_in": num_features,
            "seed_value": seed_value,
            # "callbacks": [early_stopping],
        }

        dict_param_grid_regressor = {
            'optimizer': ['rmsprop'],
            'learning_rate': [1E-4, 1E-5, 1E-6, 1E-7],
            'hidden_layer_sizes': [(num_features,)]
            # 'hidden_layer_sizes': [(num_features, 6, 2), (num_features, 6, 4), (num_features, 6, 8),
            #                        (num_features, 6, 10), (num_features, 6, 12), (num_features, 6, 14), (num_features, 6, 16)]
            # 'hidden_layer_sizes': [(num_features, 2), (num_features, 4), (num_features, 6), (num_features, 8),
            #                        (num_features, 10), (num_features, 12), (num_features, 14), (num_features, 16),
            #                        (num_features, 18), (num_features, 20), (num_features, 22), (num_features, 24)]
        }

        regressor = MLPKerasRegressor(**fit_kwargs)

    # logger.info('Hyperparameters of regressor {}: {}'.format(regressor_name, str(dict_param_grid_regressor)))

    list_dicts_params_learning_curve = []

    for key, value in dict_param_grid_regressor.items():
        dict_param_lrc = {'param_name': key, 'param_range': value}
        list_dicts_params_learning_curve.append(dict_param_lrc)

    return regressor, dict_param_grid_regressor, list_dicts_params_learning_curve


def get_fs_method(fs_method_name, estimator_name, n_features, seed_value):
    if fs_method_name == 'relief':
        return Relief(return_scores=True)
    elif fs_method_name == 'surf':
        return Surf(return_scores=True)
    elif fs_method_name == 'mrmr':
        return MRMR(return_scores=True)
    elif fs_method_name == 'pi':
        estimator, dict_param_grid_regressor, _ = get_hyperparams(estimator_name, num_features=n_features)
        return PermutationImportance(return_scores=True, seed_value=seed_value,
                                     estimator=estimator, param_grid_estimator=dict_param_grid_regressor)
    elif fs_method_name == 'relief':
        logger.info('Training with relief')
        return Relief(return_scores=True)


def save_feature_scores_by_seed(df_feature_scores, estimator_name, fs_method_name, type_over, seed_value):
    logger.info('Saving feature scores, estimator {}, seed {}'.format(estimator_name, seed_value))
    dict_scores = dict(zip(df_feature_scores.var_name, df_feature_scores.score))
    dict_scores['date'] = datetime.datetime.now()
    dict_scores['type_over'] = type_over
    dict_scores['seed'] = seed_value
    dict_scores['estimator'] = estimator_name
    dict_scores['fs_method'] = fs_method_name

    df_importance_values = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_METRICS, 'importance_values.csv')))
    df_importance_values = df_importance_values.append(dict_scores, ignore_index=True)
    df_importance_values.to_csv(str(Path.joinpath(consts.PATH_PROJECT_METRICS, 'importance_values.csv')), index=False)
    print(df_importance_values)


def train_fs_with_different_k_features(x_features, y_label, v_col_names, fs_method_name, estimator_name,
                                       list_vars_numerical, list_vars_categorical, join_synthetic=False,
                                       type_over='wo'):

    df_metrics = pd.DataFrame(columns=['seed', 'estimator', 'fs_method', 'k_features', 'mae', 'mrae'])
    list_dict_metrics = []

    for idx in np.arange(1, 6, 1):

        seed = np.random.seed(idx)
        rng = np.random.RandomState(idx)

        x_train, x_test, y_train, y_test = train_test_split(x_features, y_label, test_size=TEST_SIZE_PROPORTION, random_state=idx)

        print(pd.DataFrame(x_train, columns=v_col_names))

        x_train, y_train, x_test, y_test, v_col_names_ohe = encode_albuminuria(x_train, y_train, x_test, y_test, v_col_names)

        if join_synthetic and type_over != 'wo':
            logger.info('Joining real and synthetic data with {}'.format(type_over))
            x_train, y_train = join_real_synthetic_samples(x_train, y_train,
                                                           type_oversampling=type_over,
                                                           seed_value=idx)

        x_train, x_test, df_x_train_raw, df_x_test_raw = scale_features(x_train, x_test, v_col_names_ohe, list_vars_numerical, list_vars_categorical)
        df_x_train_features = pd.DataFrame(x_train, columns=v_col_names_ohe)
        df_x_test_features = pd.DataFrame(x_test, columns=v_col_names_ohe)

        for k_features in range(1, df_x_train_features.shape[1] + 1):
            logger.info('Training with {}, k_features: {}'.format(estimator_name, k_features))
            fs_method = get_fs_method(fs_method_name, estimator_name, n_features=k_features, seed_value=idx)
            fs_method.fit(df_x_train_features, y_train, k_features, list_vars_categorical, list_vars_numerical)
            df_filtered, df_feature_scores = fs_method.extract_features()

            v_selected_features = df_filtered.columns.values
            df_x_train_selected_features = df_x_train_features.loc[:, v_selected_features]
            df_x_test_selected_features = df_x_test_features.loc[:, v_selected_features]

            logger.info(
                'seed: {}, {}, fs: {}, features: {}'.format(idx, estimator_name, fs_method_name, v_selected_features))

            model_regressor, param_grid_regressor, list_dict_params = get_hyperparams(estimator_name,
                                                                                      df_x_train_selected_features.shape[1],
                                                                                      seed_value=idx)
            grid_cv = GridSearchCV(estimator=model_regressor,
                                   param_grid=param_grid_regressor,
                                   cv=5,
                                   return_train_score=True,
                                   scoring=make_scorer(compute_mrae, greater_is_better=False)
                                   )

            grid_cv.fit(df_x_train_selected_features.values, y_train)
            best_clf = grid_cv.best_estimator_
            y_pred = best_clf.predict(df_x_test_selected_features.values)

            mae = mean_absolute_error(y_test, y_pred)
            mrae = compute_mrae(y_test, y_pred)

            list_dict_metrics.append({'seed': idx,
                                      'estimator': estimator_name,
                                      'fs_method': fs_method_name,
                                      'k_features': k_features,
                                      'mae': mae,
                                      'mrae': mrae
            })

            logger.info('seed: {}, {}, fs: {}, mae: {}, mrae: {}'.format(idx, estimator_name, fs_method_name, mae, mrae))

        save_feature_scores_by_seed(df_feature_scores, estimator_name, fs_method_name, type_over, seed_value=idx)

    df_metrics = df_metrics.append(list_dict_metrics, ignore_index=True)

    df_metrics_average = pd.DataFrame(columns=['estimator', 'fs_method', 'k_features', 'metric', 'mean', 'std'])

    list_dict_metrics_average = []
    for k_features in np.arange(1, 13, 1):
        df_seed = df_metrics[df_metrics['k_features'] == k_features]

        list_dict_metrics_average.append({'estimator': estimator_name,
                                          'fs_method': fs_method_name,
                                          'k_features': k_features,
                                          'metric': 'mae',
                                          'mean': df_seed['mae'].mean(),
                                          'std': df_seed['mae'].std()
                                          })

        list_dict_metrics_average.append({'estimator': estimator_name,
                                          'fs_method': fs_method_name,
                                          'k_features': k_features,
                                          'metric': 'mrae',
                                          'mean': df_seed['mrae'].mean(),
                                          'std': df_seed['mrae'].std()
                                          })

    df_metrics_average = df_metrics_average.append(list_dict_metrics_average, ignore_index=True)

    return df_metrics_average


def evaluate_fs_method(fs_method_name, estimator_name,
                       df_features, y_label, k_features,
                       list_vars_numerical, list_vars_categorical):

    num_features = df_features.shape[1]
    regressor, param_grid_regressor, _ = get_hyperparams(estimator_name, num_features=num_features)

    fs_method = get_fs_method(fs_method_name, estimator_name, n_features=k_features, seed_value=idx)
    fs_method.fit(df_features, y_label, k_features, list_vars_categorical, list_vars_numerical)
    df_filtered, df_feature_scores = fs_method.extract_features()

    fsm = PermutationImportance(return_scores=True, seed_value=k_features, estimator=regressor, param_grid_estimator=param_grid_regressor)
    fsm.fit(df_features, y_label, k_features, list_vars_categorical, list_vars_numerical)
    df_filtered, df_feature_scores = fsm.extract_features()
    # df_metrics = train_fs_with_different_k_features(df_filtered, y_label, fs_method_name, regressor_name,
    #                                    list_categorical_vars, list_numerical_vars)
    # print(df_feature_scores)
    # print(df_metrics)
    print(df_filtered)
    plot_fs_score(df_feature_scores, fs_method_name, k_features, estimator_name,
                  flag_save_figure=True)



# def train_with_several_partitions(x_features, y_label, regressor_name, fs_method_name, k_features,
#                                   plot_learning_curve=False, as_frame=False, scoring='roc_auc',
#                                   verbose=False, flag_save_figure=False):
#
#     list_mae_vals = []
#     list_mrae_vals = []
#
#     for idx in np.arange(1, 6, 1):
#
#         seed = np.random.seed(idx)
#         rng = np.random.RandomState(idx)
#
#         x_train, x_test, y_train, y_test = train_test_split(x_features, y_label, test_size=TEST_SIZE_PROPORTION,
#                                                             random_state=idx)
#
#         num_features = x_features.shape[1]
#
#         model_regressor, param_grid_regressor, list_dict_params = get_hyperparams(regressor_name, num_features, seed_value=idx)
#
#         dict_params_gridsearch = {
#             'regressor': model_regressor,
#             'param_grid': param_grid_regressor,
#             'cv': 3,
#             'scoring': make_scorer(compute_mrae, greater_is_better=False)
#         }
#
#         grid_cv = GridSearchCV(estimator=dict_params_gridsearch['regressor'],
#                                param_grid=dict_params_gridsearch['param_grid'],
#                                cv=dict_params_gridsearch['cv'],
#                                return_train_score=True,
#                                scoring=dict_params_gridsearch['scoring']
#                                )
#
#         grid_cv.fit(x_train, y_train)
#         best_clf = grid_cv.best_estimator_
#         y_pred = best_clf.predict(x_test)
#
#         grid_best_params = grid_cv.best_params_
#
#         if plot_learning_curve:
#             plot_learning_curves_several_hyperparameters(x_train, y_train, regressor_name, model_regressor,
#                                                          seed_value=idx,
#                                                          list_dict_params=list_dict_params,
#                                                          cv=dict_params_gridsearch['cv'],
#                                                          scoring=make_scorer(compute_mrae, greater_is_better=True),
#                                                          grid_best_params=grid_best_params,
#                                                          flag_save_figure=flag_save_figure)
#
#         mae = mean_absolute_error(y_test, y_pred)
#         mrae = compute_mrae(y_test, y_pred)
#
#         logger.info('Partition {}, best hyperparams: {}'.format(idx, grid_best_params))
#         logger.info('Partition {}, regressor: {}, fs: {}, k: {}, mae: {}'.format(idx, regressor_name, fs_method_name, k_features, mae))
#         logger.info('Partition {}, regressor: {}, fs: {}, k: {}, mrae: {}'.format(idx, regressor_name, fs_method_name, k_features, mrae))
#
#         list_mae_vals.append(mae)
#         list_mrae_vals.append(mrae)
#
#     mean_mae = np.mean(np.array(list_mae_vals))
#     std_mae = np.std(np.array(list_mae_vals))
#     mean_mrae = np.mean(np.array(list_mrae_vals))
#     std_mrae = np.std(np.array(list_mrae_vals))
#
#     list_dict_fm = [{'regressor': regressor_name,
#                      'fs_method': fs_method_name,
#                      'k_features': k_features,
#                      'metric': 'mae',
#                      'mean': mean_mae,
#                      'std': std_mae
#                      }, {'regressor': regressor_name,
#                          'fs_method': fs_method_name,
#                          'k_features': k_features,
#                          'metric': 'mrae',
#                          'mean': mean_mrae,
#                          'std': std_mrae
#                          }]
#
#     logger.info('Regressor {}, mae: {}({})'.format(regressor_name, mean_mae, std_mae))
#     logger.info('Regressor {}, mrae: {}({})'.format(regressor_name, mean_mrae, std_mrae))
#
#     return list_dict_fm
