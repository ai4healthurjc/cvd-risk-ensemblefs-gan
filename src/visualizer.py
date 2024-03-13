import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn_genetic import GAFeatureSelectionCV
from pathlib import Path
import datetime
import logging
import coloredlogs
import shap
from zoofs import ParticleSwarmOptimization
import joblib
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


parser = argparse.ArgumentParser(description='Process representations.')
args = parse_arguments(parser)

x_features, y_label, v_col_names, list_vars_categorical, list_vars_numerical = load_preprocessed_dataset(consts.BBDD_STENO, show_info=False)
df_features = pd.DataFrame(x_features, columns=v_col_names)

print(df_features)

x_features = df_features.values
list_vars_categorical.remove('album')
list_vars_categorical.extend(['album_0', 'album_1', 'album_2'])

