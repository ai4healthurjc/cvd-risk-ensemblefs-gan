import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from pathlib import Path
import datetime
import logging
import coloredlogs
import shap
import joblib
from zoofs import ParticleSwarmOptimization

from estimators.regression import get_hyperparams, train_fs_with_different_k_features, scale_features, encode_albuminuria, get_fs_method
from utils.loader import load_preprocessed_dataset, save_model_estimator, load_model_estimator
from utils.quality import generate_concatenate_real_synthetic_samples, join_real_synthetic_samples
from utils.metrics import compute_mrae, compute_mae
from utils.plotter import plot_learning_curve_mlp, plot_learning_curves_several_hyperparameters, \
    plot_scatter_real_pred, plot_hists_comparison, plot_ale_features
import utils.consts as consts


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


list_regression_metrics = ['mean_squared_error', 'mean_absolute_error', 'compute_mrae']
list_features_fs2 = ['age', 'sex', 'smoking', 'exercise', 'dm_duration']


def encode_categorical_vars(df, var, target):
    df_temp = df[[var, target]].copy()
    t = df_temp.groupby(var).median().sort_values(target, ascending=True).index
    cat_encoder_dict = {k: i for i, k in enumerate(t, 0)}
    df[var] = df[var].map(cat_encoder_dict)
    return df


def perform_fs_method(fes,
                      x_train,
                      y_train,
                      x_test,
                      y_test,
                      df_x_test_raw,
                      v_col_names,
                      estimator_name,
                      type_over,
                      list_vars_numerical,
                      list_vars_categorical,
                      seed_value
                      ):

    df_x_train = pd.DataFrame(x_train, columns=v_col_names)
    df_x_test = pd.DataFrame(x_test, columns=v_col_names)
    v_complete_features = v_col_names
    v_selected_features = v_col_names

    logger.info('Perform fs with {} and estimator {}'.format(fes, estimator_name))

    if fes == 'fes1':
        v_selected_features = v_selected_features
    elif fes == 'fes2':
        v_selected_features = np.array(list_features_fs2)
    elif fes == 'fes3':
        if estimator_name == 'mlp':
            v_selected_features = ['age', 'album_0', 'hba1c', 'sex', 'exercise', 'album_1', 'sbp', 'smoking', 'ldl',
                                   'dm_duration', 'egfr']
        else:
            v_selected_features = perform_fs_by_method(x_train, y_train, v_col_names, estimator_name, 'pi', type_over,
                                                       list_vars_numerical, list_vars_categorical, seed_value)

    elif fes == 'fes4':
        v_selected_features = perform_pso_fs(x_train, y_train, v_col_names, estimator_name)
    elif fes == 'fes5':
        v_selected_features = perform_fs_by_method(x_train, y_train, v_col_names,
                                                   estimator_name, 'mrmr', type_over,
                                                   list_vars_numerical, list_vars_categorical,
                                                   seed_value)

    elif fes == 'fes6':
        v_selected_features = perform_fs_by_method(x_train, y_train, v_col_names,
                                                   estimator_name, 'relief', type_over,
                                                   list_vars_numerical, list_vars_categorical,
                                                   seed_value)

    logger.info('fes: {}, selected-features: {}'.format(fes, v_selected_features))

    df_x_train_filtered = df_x_train.loc[:, v_selected_features]
    df_x_test_filtered = df_x_test.loc[:, v_selected_features]
    df_x_test_raw_filtered = df_x_test_raw.loc[:, v_selected_features]

    v_removed_features = np.setdiff1d(v_complete_features, v_selected_features)

    save_selected_features(v_selected_features, v_removed_features, fes, estimator_name, type_over, seed_value)

    return df_x_train_filtered, y_train, df_x_test_filtered, y_test, v_removed_features, df_x_test_raw_filtered


def perform_fs_by_method(x_train,
                         y_train,
                         v_col_names,
                         estimator_name,
                         fs_method_name,
                         type_over,
                         list_vars_numerical,
                         list_vars_categorical,
                         seed_value=4242
                         ):

    total_features = x_train.shape[1]
    df_x_train = pd.DataFrame(x_train, columns=v_col_names)
    df_x_test = pd.DataFrame(x_test, columns=v_col_names)

    list_mae = []
    list_mrae = []

    k_features_range = np.arange(1, total_features + 1)

    for k_features in k_features_range:
        fs_method = get_fs_method(fs_method_name, estimator_name, n_features=k_features, seed_value=seed_value)
        fs_method.fit(df_x_train, y_train, k_features, list_vars_categorical, list_vars_numerical)
        df_x_train_selected_features, df_feature_scores = fs_method.extract_features()
        df_x_test_selected_features = df_x_test.loc[:, df_x_train_selected_features.columns.values]

        model_estimator, param_grid_estimator, _ = get_hyperparams(estimator_name,
                                                                   df_x_train_selected_features.shape[1],
                                                                   seed_value=seed_value)
        grid_cv = GridSearchCV(estimator=model_estimator,
                               param_grid=param_grid_estimator,
                               cv=5,
                               return_train_score=True,
                               n_jobs=-1,
                               scoring=make_scorer(compute_mrae, greater_is_better=False)
                               )

        grid_cv.fit(df_x_train_selected_features.values, y_train)
        best_clf = grid_cv.best_estimator_
        y_pred = best_clf.predict(df_x_test_selected_features.values)

        list_mae.append(mean_absolute_error(y_test, y_pred))
        list_mrae.append(compute_mrae(y_test, y_pred))

    v_mae = np.array(list_mae)
    v_mrae = np.array(list_mrae)
    v_diff_mrae = np.diff(v_mrae)

    index_min_mrae = np.argmin(v_mrae)
    k_optimal = k_features_range[index_min_mrae]
    v_selected_features = df_feature_scores['var_name'].values[:k_optimal]

    # print('mae: ', v_mae)
    # print('mrae: ', v_mrae)
    # print('k_optimal: ', k_optimal)

    logger.info('fs-method {}, k-optimal: {}'.format(fs_method_name, k_optimal))

    return v_selected_features


def save_regression_metric(fs, regressor, type_over, mae_mean, mae_std, mrae_mean, mrae_std):

    dict_metric = {
        'date': datetime.datetime.now(),
        'fs': fs,
        'estimator': regressor,
        'type_over': type_over,
        'mae_mean': mae_mean,
        'mae_std': mae_std,
        'mrae_mean': mrae_mean,
        'mrae_std': mrae_std
    }

    path_metrics = Path.joinpath(consts.PATH_PROJECT_METRICS, 'metrics_regression.csv')
    if path_metrics.exists():
        df_metrics_regression = pd.read_csv(str(path_metrics))
        df_metrics_regression = pd.concat([df_metrics_regression, pd.DataFrame([dict_metric])], ignore_index=True)
        df_metrics_regression.to_csv(str(path_metrics), index=False)
    else:
        df_metrics_regression = pd.DataFrame([dict_metric])
        df_metrics_regression.to_csv(str(path_metrics), index=False)


def save_selected_features(v_selected_features, v_non_selected_features, fs_method_name, estimator_name, type_over, seed_value):

    dict_selected_features = {x: 1 for x in v_selected_features}
    dict_non_selected_features = {x: 0 for x in v_non_selected_features}

    dict_features = {
        'date': datetime.datetime.now(),
        'fs_method': fs_method_name,
        'estimator': estimator_name,
        'type_over': type_over,
        'seed': seed_value
    }

    dict_features.update(dict_selected_features)
    dict_features.update(dict_non_selected_features)

    path_selected_features = Path.joinpath(consts.PATH_PROJECT_METRICS, 'selected_features.csv')

    if path_selected_features.exists():
        df_selected_features = pd.read_csv(str(path_selected_features))
        df_selected_features = pd.concat([df_selected_features, pd.DataFrame([dict_features])], ignore_index=True)
        df_selected_features.to_csv(str(path_selected_features), index=False)
    else:
        df_selected_features = pd.DataFrame([dict_features])
        df_selected_features.to_csv(str(path_selected_features), index=False)


def objective_function_topass(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    P = compute_mrae(y_valid, model.predict(X_valid))
    return P


def perform_pso_fs(x_train, y_train, v_col_names, estimator_name) -> np.array:
    model_regressor, param_grid_regressor, list_dict_params = get_hyperparams(estimator_name, x_train.shape[1])
    df_x_train = pd.DataFrame(x_train, columns=v_col_names)
    df_x_test = pd.DataFrame(x_test, columns=v_col_names)
    algo_object = ParticleSwarmOptimization(objective_function_topass,
                                            n_iteration=25,
                                            population_size=20,
                                            minimize=True
                                            )
    algo_object.fit(model_regressor, df_x_train, y_train, df_x_test, y_test, verbose=True)
    v_selected_features = algo_object.best_feature_list

    return v_selected_features


# def perform_genetic_fs(x_train, y_train, v_col_names, estimator_name) -> np.array:
#
#     model_regressor, param_grid_regressor, list_dict_params = get_hyperparams(estimator_name, x_train.shape[1])
#     df_x_train = pd.DataFrame(x_train, columns=v_col_names)

    # ga_model = GeneticSelectionCV(estimator=model_regressor, cv=5, verbose=1,
    #                               scoring=make_scorer(compute_mrae, greater_is_better=False),
    #                               max_features=12, n_population=300, crossover_proba=0.5,
    #                               mutation_proba=0.2, n_generations=50,
    #                               crossover_independent_proba=0.5,
    #                               mutation_independent_proba=0.04,
    #                               tournament_size=3, n_gen_no_change=10,
    #                               caching=True, n_jobs=-1)

    # ga_model = ga_model.fit(x_train, y_train)
    # v_selected_features = df_x_train.loc[:, ga_model.support_].columns.values

    # ga_model = GAFeatureSelectionCV(
    #     estimator=model_regressor,
    #     cv=5,
    #     scoring=make_scorer(compute_mrae, greater_is_better=False),
    #     population_size=30,
    #     generations=20,
    #     n_jobs=-1,
    #     verbose=True,
    #     algorithm='eaSimple',
    #     keep_top_k=2,
    #     elitism=True,
    # )
    #
    # ga_model.fit(x_train, y_train)
    # v_selected_features_idx = ga_model.best_features_
    # v_selected_features = df_x_train.loc[:, v_selected_features_idx].columns.values
    #
    # return v_selected_features


def decode_ohe(m_ohe):
    m_decoded = np.argmax(m_ohe, axis=1)
    return m_decoded


def parse_arguments(parser):
    parser.add_argument('--estimator', default='dt', type=str)
    parser.add_argument('--fes', default='fes4', type=str)
    parser.add_argument('--type_over', default='wo', type=str)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--plot_ale', default=False, type=bool)
    parser.add_argument('--remove_age', default=False, type=bool)
    parser.add_argument('--plot_scatter_hists', default=False, type=bool)
    parser.add_argument('--flag_save_figure', default=True, type=bool)
    parser.add_argument('--train_shap', default=False, type=bool)
    parser.add_argument('--generate_synthetic', default=False, type=bool)
    parser.add_argument('--join_synthetic', default=False, type=bool)
    parser.add_argument('--plot_learning_curve', default=False, type=bool)
    return parser.parse_args()


def save_train_test_partition_data(df_x_train, y_train, df_x_test, y_test, fes, type_over, estimator, seed_value):

    df_x_train_copy = df_x_train.copy()
    df_x_test_copy = df_x_test.copy()

    df_x_train_copy['label'] = y_train
    df_x_test_copy['label'] = y_test

    df_x_train_copy.to_csv(
        str(Path.joinpath(consts.PATH_PROJECT_DATA_PARTITIONS,
                          'df_x_train_{}_{}_{}_{}.csv'.format(fes, type_over, estimator, seed_value))),
        index=False
    )

    df_x_test_copy.to_csv(
        str(Path.joinpath(consts.PATH_PROJECT_DATA_PARTITIONS,
                          'df_x_test_{}_{}_{}_{}.csv'.format(fes, type_over, estimator, seed_value))),
        index=False
    )


def train_shap_explainer(estimator,
                         fes,
                         type_over,
                         estimator_name,
                         df_x_train,
                         y_train,
                         df_x_test,
                         y_test,
                         seed_value,
                         save_shap=False
                         ):

    save_train_test_partition_data(df_x_train, y_train, df_x_test, y_test, fes, type_over, estimator_name, seed_value)

    filename_pattern = '{}_{}_{}_{}'.format(fes, type_over, estimator_name, seed_value)
    path_shap_explainer = str(Path.joinpath(consts.PATH_PROJECT_INTERPRETER, 'shap_{}'.format(filename_pattern)))

    if estimator_name == 'dt' or estimator_name == 'rf':
        explainer = shap.TreeExplainer(estimator, df_x_train)
    else:
        explainer = shap.KernelExplainer(estimator.predict, df_x_train)

    shap_values = explainer.shap_values(df_x_test)
    feature_names = df_x_train.columns

    vals = np.abs(pd.DataFrame(shap_values, columns=feature_names).values).mean(0)
    df_shap = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance'])
    df_shap_importance = df_shap.set_index('col_name')
    df_shap_row = df_shap_importance.T
    df_shap_row['fes'] = fes
    df_shap_row['type_over'] = type_over
    df_shap_row['estimator'] = estimator_name
    df_shap_row['seed'] = seed_value

    path_shap_feature_importances = Path.joinpath(consts.PATH_PROJECT_METRICS, 'shap_importances.csv')

    if path_shap_feature_importances.exists():
        df_shap_total = pd.read_csv(str(path_shap_feature_importances))
        df_shap_total = df_shap_total.append(df_shap_row)
    else:
        df_shap_total = df_shap_row

    df_shap_total.to_csv(str(path_shap_feature_importances), index=False)
    path_shap_values = Path.joinpath(consts.PATH_PROJECT_INTERPRETER, 'shap_test_values_{}.npy'.format(filename_pattern))

    if save_shap:
        joblib.dump(explainer, filename='{}.{}'.format(path_shap_explainer, 'bz2'), compress=('bz2', 9))
        np.save(str(path_shap_values), shap_values)


def get_best_estimator(estimator, df_x_train, y_train, seed_value):

    n_features = df_x_train.shape[1]

    model_estimator, param_grid_estimator, list_dict_params = get_hyperparams(estimator,
                                                                              num_features=n_features,
                                                                              seed_value=seed_value)

    grid_cv = GridSearchCV(estimator=model_estimator,
                           param_grid=param_grid_estimator,
                           cv=5,
                           return_train_score=True,
                           scoring=make_scorer(compute_mrae, greater_is_better=False),
                           n_jobs=-1
                           )

    grid_cv.fit(df_x_train.values, y_train)
    best_estimator = grid_cv.best_estimator_

    return best_estimator, grid_cv.best_params_, model_estimator, list_dict_params


parser = argparse.ArgumentParser(description='Process representations.')
args = parse_arguments(parser)

x_features, y_label, v_col_names, list_vars_categorical, list_vars_numerical = load_preprocessed_dataset(consts.BBDD_STENO, show_info=False)
df_features = pd.DataFrame(x_features, columns=v_col_names)

x_features = df_features.values
list_vars_categorical.remove('album')
list_vars_categorical.extend(['album_0', 'album_1', 'album_2'])

list_mae = []
list_mrae = []

for idx in np.arange(1, 6, 1):

    seed = np.random.seed(idx)
    rng = np.random.RandomState(idx)

    x_train, x_test, y_train, y_test = train_test_split(x_features, y_label, test_size=consts.TEST_SIZE_PROPORTION, random_state=idx)

    if args.generate_synthetic:
        generate_concatenate_real_synthetic_samples(x_train, y_train,
                                                    v_col_names, idx,
                                                    args.type_over,
                                                    save_synthetic_data=False,
                                                    train_ctgan=False,
                                                    n_epochs=200,
                                                    batch_size=32,
                                                    flag_cuda=False,
                                                    )

    x_train, y_train, x_test, y_test, v_col_names_ohe = encode_albuminuria(x_train, y_train, x_test, y_test, v_col_names)

    if args.join_synthetic and args.type_over != 'wo':
        x_train, y_train = join_real_synthetic_samples(x_train,
                                                       y_train,
                                                       type_oversampling=args.type_over,
                                                       seed_value=idx)

    x_train, x_test, df_x_train_raw, df_x_test_raw = scale_features(x_train,
                                                                    y_train,
                                                                    x_test,
                                                                    y_test,
                                                                    v_col_names_ohe,
                                                                    list_vars_numerical,
                                                                    list_vars_categorical,
                                                                    seed_value=idx,
                                                                    save_data=False
                                                                    )

    df_x_train, y_train, df_x_test, y_test, v_removed_features, df_x_test_raw = perform_fs_method(args.fes,
                                                                                   x_train, y_train,
                                                                                   x_test, y_test,
                                                                                   df_x_test_raw,
                                                                                   v_col_names_ohe,
                                                                                   args.estimator,
                                                                                   args.type_over,
                                                                                   list_vars_numerical,
                                                                                   list_vars_categorical,
                                                                                   seed_value=idx
                                                                                   )

    if args.remove_age:
        # print(df_x_train)
        df_x_train = df_x_train.drop(columns=['age'])
        df_x_test = df_x_test.drop(columns=['age'])
        df_x_test_raw = df_x_test_raw.drop(columns=['age'])
        v_removed_features = np.concatenate((v_removed_features, np.array(['age'])))

    best_estimator, best_grid_params, model_estimator, list_dict_params = get_best_estimator(args.estimator, df_x_train, y_train, idx)
    y_pred = best_estimator.predict(df_x_test.values)

    save_model_estimator(best_estimator, args.fes, args.type_over, args.estimator, idx)
    save_train_test_partition_data(df_x_train, y_train, df_x_test, y_test, args.fes, args.type_over, args.estimator, seed_value=idx)

    if args.train_shap:
        train_shap_explainer(best_estimator,
                             args.fes,
                             args.type_over,
                             args.estimator,
                             df_x_train,
                             y_train,
                             df_x_test,
                             y_test,
                             seed_value=idx,
                             save_shap=True
                             )

    if args.plot_ale:
        plot_ale_features(df_x_test, v_removed_features, best_estimator, df_x_test_raw,
                          list_vars_numerical, list_vars_categorical,
                          args.estimator, args.fes, args.type_over, seed_value=idx)

    if args.estimator == 'mlp':
        plot_learning_curve_mlp(best_estimator.history_,
                                args.estimator,
                                idx,
                                list_regression_metrics,
                                flag_save_figure=args.flag_save_figure)

    if args.plot_learning_curve:

        plot_learning_curves_several_hyperparameters(x_train, y_train, args.estimator, model_estimator,
                                                     seed_value=idx,
                                                     list_dict_params=list_dict_params,
                                                     cv=5,
                                                     scoring=make_scorer(compute_mrae, greater_is_better=True),
                                                     grid_best_params=best_grid_params,
                                                     flag_save_figure=args.flag_save_figure)

    mae = compute_mae(y_test, y_pred)
    mrae = compute_mrae(y_test, y_pred)

    logger.info('Partition {}, best hyperparams: {}'.format(idx, best_grid_params))
    logger.info('Partition {}, mae: {}'.format(idx, mae))
    logger.info('Partition {}, mrae: {}'.format(idx, mrae))

    list_mae.append(mae)
    list_mrae.append(mrae)

    if args.plot_scatter_hists:
        plot_scatter_real_pred(y_test, y_pred,
                               # title_figure='mae: {}, mrae: {}'.format(round(mae, 4), round(mrae, 4)),
                               title_figure='',
                               title_file=args.estimator,
                               seed_value=idx, flag_save_figure=args.flag_save_figure)

        plot_hists_comparison(y_test, y_pred,
                              title_file=args.estimator,
                              seed_value=idx, flag_save_figure=args.flag_save_figure)

mae_mean = np.mean(np.array(list_mae))
mae_std = np.std(np.array(list_mae))
mrae_mean = np.mean(np.array(list_mrae))
mrae_std = np.std(np.array(list_mrae))

logger.info('Average results for regressor {}, mae: {}({})'.format(args.estimator, mae_mean, mae_std))
logger.info('Average results for regressor {}, mrae: {}({})'.format(args.estimator, mrae_mean, mrae_std))

save_regression_metric(args.fes, args.estimator, args.type_over, mae_mean, mae_std, mrae_mean, mrae_std)

