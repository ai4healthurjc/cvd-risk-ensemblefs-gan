import numpy as np
import pandas as pd
import argparse
from utils.loader import load_preprocessed_dataset
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
import random as python_random
import utils.consts as consts
from estimators.regression import compute_evolution_k_parameters
import logging
import coloredlogs
from estimators.regression import get_hyperparams, train_fs_with_different_k_features, evaluate_fs_method, encode_albuminuria, scale_features
from utils.metrics import compute_mrae, compute_mae
from utils.quality import generate_concatenate_real_synthetic_samples, compute_quality_metrics, join_real_synthetic_samples
from utils.metrics import compute_mrae, compute_mae, compute_mse
from utils.plotter import plot_learning_curve_mlp, plot_learning_curves_several_hyperparameters, \
    plot_scatter_real_pred, plot_hists_comparison, plot_gridsearch_results, \
    plot_performance_evolution_k_features, plot_quality_scores, plot_area_cis_average, plot_heatmap_selected_features_fes,\
    plot_fs_score, plot_shap_mean, plot_scatter_from_model


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

TOT = 1e-4
MAX_ITERS = 20000
TEST_SIZE_PROPORTION = 0.2
VALIDATION_SIZE_PROPORTION = 0.2
list_regression_metrics = ['mean_squared_error', 'mean_absolute_error', 'compute_mrae']


def encode_categorical_vars(df, var, target):
    df_temp = df[[var, target]].copy()
    t = df_temp.groupby(var).median().sort_values(target, ascending=True).index
    cat_encoder_dict = {k: i for i, k in enumerate(t, 0)}
    df[var] = df[var].map(cat_encoder_dict)
    return df


def parse_arguments(parser):
    parser.add_argument('--estimator', default='mlp_keras', type=str)
    parser.add_argument('--fs', default='fes3', type=str)
    parser.add_argument('--type_over', default='wo', type=str)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--flag_save_figure', default=True, type=bool)
    parser.add_argument('--train_ctgan', default=False, type=bool)
    parser.add_argument('--cuda', default=False, type=bool)
    parser.add_argument('--join_synthetic', default=False, type=bool)
    parser.add_argument('--plot_learning_curve', default=False, type=bool)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='Process representations.')
args = parse_arguments(parser)

x_features, y_label, v_col_names, list_vars_categorical, list_vars_numerical = load_preprocessed_dataset(consts.BBDD_STENO, show_info=False)
df_features = pd.DataFrame(x_features, columns=v_col_names)
list_vars_categorical.remove('album')
list_vars_categorical.extend(['album_0', 'album_1', 'album_2'])

plot_scatter_from_model(args.fs, args.type_over, args.estimator, seed_value=1)
plot_scatter_from_model(args.fs, args.type_over, args.estimator, seed_value=2)
plot_scatter_from_model(args.fs, args.type_over, args.estimator, seed_value=3)
plot_scatter_from_model(args.fs, args.type_over, args.estimator, seed_value=4)
plot_scatter_from_model(args.fs, args.type_over, args.estimator, seed_value=5)

# plot_area_cis_average(args.fs, args.estimator, flag_save_figure=True)
#
# plot_heatmap_selected_features_fes(args.fs, args.type_over, flag_save_figure=True)

# plot_shap_mean(args.fs, args.type_over, args.estimator, flag_save_figure=True)

# compute_evolution_k_parameters(df_features, y_label, v_col_names,
#                                args.estimator, 'mrmr', args.type_over, args.join_synthetic,
#                                list_vars_numerical, list_vars_categorical)

# evaluate_fs_method(args.fs, args.regressor, df_features, y_label, 5, list_vars_categorical, list_vars_numerical)

# plot_fs_score(df_feature_scores, fs_method_name, k_features, estimator_name, flag_save_figure=True)

list_mae = []
list_mrae = []
list_quality_total = []

for idx in np.arange(1, 6, 1):

    seed = np.random.seed(idx)
    rng = np.random.RandomState(idx)

    x_train, x_test, y_train, y_test = train_test_split(x_features,
                                                        y_label,
                                                        test_size=TEST_SIZE_PROPORTION,
                                                        random_state=idx)

    x_train_resampled, y_train_resampled = generate_concatenate_real_synthetic_samples(x_train,
                                                                                   y_train,
                                                                                   v_col_names,
                                                                                   idx,
                                                                                   args.type_over,
                                                                                   list_vars_categorical,
                                                                                   train_ctgan=args.train_ctgan,
                                                                                   n_epochs=200,
                                                                                   batch_size=40,
                                                                                   flag_cuda=args.cuda
                                                                                   )

    if args.join_synthetic and args.type_over != 'wo':
        x_train, y_train = join_real_synthetic_samples(x_train, y_train,
                                                       type_oversampling=args.type_over,
                                                       seed_value=idx)

    x_train, y_train, x_test, y_test, v_col_names_ohe = encode_albuminuria(x_train,
                                                                           y_train,
                                                                           x_test,
                                                                           y_test,
                                                                           v_col_names)

    x_train, x_test, df_x_train_raw, df_x_test_raw = scale_features(x_train, x_test, v_col_names_ohe,
                                                                    list_vars_numerical, list_vars_categorical)

    print('xxxxxx')
    print('x_train_resampled: ', x_train_resampled.shape)
    print('x_train: ', x_train.shape)
    print('xxxxxx')

    # list_quality_total.extend(list_qscores)
    # print(pd.DataFrame(x_train_resampled, columns=v_col_names))

    model_regressor, param_grid_regressor, list_dict_params = get_hyperparams(args.estimator,
                                                                              x_train_resampled.shape[1],
                                                                              seed_value=idx)

    dict_params_gridsearch = {
        'regressor': model_regressor,
        'param_grid': param_grid_regressor,
        'cv': 5,
        'scoring': make_scorer(compute_mrae, greater_is_better=False)
    }

    grid_cv = GridSearchCV(estimator=dict_params_gridsearch['regressor'],
                           param_grid=dict_params_gridsearch['param_grid'],
                           cv=dict_params_gridsearch['cv'],
                           return_train_score=True,
                           scoring=dict_params_gridsearch['scoring']
                           )

    grid_cv.fit(x_train, y_train)
    best_clf = grid_cv.best_estimator_
    y_pred = best_clf.predict(x_test)

    if args.estimator == 'mlp':
        history = best_clf.history_
        plot_learning_curve_mlp(history, args.estimator, idx, list_regression_metrics,
                                flag_save_figure=args.flag_save_figure)

    grid_best_params = grid_cv.best_params_

    if args.plot_learning_curve:
        plot_learning_curves_several_hyperparameters(x_train, y_train, args.estimator, model_regressor,
                                                     seed_value=idx,
                                                     list_dict_params=list_dict_params,
                                                     cv=dict_params_gridsearch['cv'],
                                                     scoring=make_scorer(compute_mrae, greater_is_better=True),
                                                     grid_best_params=grid_best_params,
                                                     flag_save_figure=args.flag_save_figure)

    mae = mean_absolute_error(y_test, y_pred)
    mrae = compute_mrae(y_test, y_pred)

    logger.info('Partition {}, best hyperparams: {}'.format(idx, grid_best_params))
    logger.info('Partition {}, mae: {}'.format(idx, mae))
    logger.info('Partition {}, mrae: {}'.format(idx, mrae))

    list_mae.append(mae)
    list_mrae.append(mrae)

    plot_scatter_real_pred(y_test, y_pred, title_figure='mae: {}, mrae: {}'.format(round(mae, 4), round(mrae, 4)),
                           title_file=args.estimator,
                           seed_value=idx, flag_save_figure=args.flag_save_figure)

    plot_hists_comparison(y_test, y_pred, title_file=args.estimator,
                          seed_value=idx, flag_save_figure=args.flag_save_figure)


df_metrics = pd.DataFrame(list_quality_total, columns=['seed', 'ir', 'score'])
df_metrics = df_metrics.drop(columns=['seed'])
df_group = df_metrics.groupby('ir').agg([np.mean, np.std]).reset_index()

df_aux_metrics = pd.DataFrame(columns=['ir', 'mean', 'std'])
df_aux_metrics['ir'] = df_group.iloc[:, 0]
df_aux_metrics['mean'] = df_group.iloc[:, 1]
df_aux_metrics['std'] = df_group.iloc[:, 2]

# plot_quality_scores(df_aux_metrics)

mean_mae = np.mean(np.array(list_mae))
std_mae = np.std(np.array(list_mae))
mean_mrae = np.mean(np.array(list_mrae))
std_mrae = np.std(np.array(list_mrae))

logger.info('Regressor {}, mae: {}({})'.format(args.estimator, mean_mae, std_mae))
logger.info('Regressor {}, mrae: {}({})'.format(args.estimator, mean_mrae, std_mrae))

