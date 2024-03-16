import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import logging
import coloredlogs
from utils.loader import load_train_test_partitions

from utils.loader import load_preprocessed_dataset, load_model_estimator, get_categorical_numerical_names
from utils.plotter import plot_ale_features, plot_all_hists_dataset
import utils.consts as consts


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


list_regression_metrics = ['mean_squared_error', 'mean_absolute_error', 'compute_mrae']
list_features_fs2 = ['age', 'sex', 'smoking', 'exercise', 'dm_duration']


def show_regression_metrics():
    df_metrics_regression = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_METRICS, 'metrics_regression.csv')))
    print(df_metrics_regression)
    gk = df_metrics_regression.groupby(['fs', 'estimator', 'type_over'])
    print(gk.apply(lambda a: a[:]))


def parse_arguments(parser):
    parser.add_argument('--estimator', default='dt', type=str)
    parser.add_argument('--fes', default='fes4', type=str)
    parser.add_argument('--type_over', default='wo', type=str)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--seed_value', default=1, type=int)
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--plot_ale', default=False, type=bool)
    parser.add_argument('--flag_save_figure', default=True, type=bool)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='Process representations.')
args = parse_arguments(parser)

x_features, y_label, v_col_names, _, _ = load_preprocessed_dataset(consts.BBDD_STENO, show_info=False)
df_features = pd.DataFrame(x_features, columns=v_col_names)
plot_all_hists_dataset(df_features, flag_save_figure=True)

show_regression_metrics()

idx = args.seed_value

filename_pattern = '{}_{}_{}_{}'.format(args.fes, args.type_over, args.estimator, idx)

seed = np.random.seed(idx)
rng = np.random.RandomState(idx)

df_x_train, y_train, df_x_test, y_test, df_x_train_raw, df_x_test_raw, _ = load_train_test_partitions(filename_pattern, seed_value=idx)

list_vars_categorical, list_vars_numerical = get_categorical_numerical_names(df_x_train, '')

model_estimator = load_model_estimator(args.fes, args.type_over,
                                       args.estimator, idx,
                                       df_x_train.values, y_train)

v_removed_features = np.array([])

plot_ale_features(df_x_test, v_removed_features, model_estimator, df_x_test_raw,
                  list_vars_numerical, list_vars_categorical,
                  args.estimator, args.fes, args.type_over, seed_value=idx)

# plot_scatter_from_model(args.fs, args.type_over, args.estimator, seed_value=idx)
