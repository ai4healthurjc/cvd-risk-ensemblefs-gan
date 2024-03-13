import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import joblib
import shap
import seaborn as sns
from shap import TreeExplainer, Explanation
from shap.plots import waterfall
from utils.loader import load_train_test_partitions
# import matplotlib
import matplotlib.pyplot as plt
import utils.consts as consts
# matplotlib.use('Agg')


def parse_arguments(parser):
    parser.add_argument('--fes', default='fes1', type=str)
    parser.add_argument('--estimator', default='dt', type=str)
    parser.add_argument('--type_over', default='wo', type=str)
    parser.add_argument('--type_cvd', default='high', type=str)
    return parser.parse_args()


def get_cvd_risk(dict_list_patients, id_patient):
    list_patients = []
    for list_values in dict_list_patients.values():
        list_patients.extend(list_patients)
    list_patients.index()


def plot_waterfall(shap_explainer, df_x_test, df_x_test_raw, y_test, filename_pattern, id_patient):

    sv = shap_explainer(df_x_test)
    exp = Explanation(sv.values,
                      sv.base_values,
                      data=df_x_test_raw.values,
                      feature_names=df_x_test.columns)
    waterfall(exp[id_patient], show=False,)

    plt.title("Patient {} with CVD risk {}".format(id_patient, y_test[id_patient]))
    plt.savefig(
        str(Path.joinpath(consts.PATH_PROJECT_INTERPRETER, 'shap_waterplot_{}_{}.png'.format(filename_pattern, id_patient))),
        bbox_inches='tight'
    )
    plt.close()


def plot_forceplot(shap_explainer, shap_values, df_x_test_raw, y_test, filename_pattern, id_patient):
    shap.force_plot(shap_explainer.expected_value,
                    shap_values[id_patient, :],
                    df_x_test_raw.iloc[id_patient, :],
                    feature_names=df_x_test_raw.columns.values,
                    # out_names='CVD risk',
                    link="identity",
                    figsize=(18, 5),
                    text_rotation=30,
                    matplotlib=True,
                    show=False
                    )

    plt.title("Patient {} with CVD risk {}".format(id_patient, y_test[id_patient]))
    plt.savefig(
        str(Path.joinpath(consts.PATH_PROJECT_INTERPRETER, 'shap_forceplot_{}_{}.png'.format(filename_pattern, id_patient))),
        bbox_inches='tight'
    )
    # plt.gcf().set_size_inches(50, 6)
    plt.tight_layout()
    plt.close()


def load_shap_model(filename_pattern):
    path_shap_explainer = str(Path.joinpath(consts.PATH_PROJECT_INTERPRETER, 'shap_{}.bz2'.format(filename_pattern)))
    shap_exp = joblib.load(filename=path_shap_explainer)
    return shap_exp


def plot_decision_plot(shap_explainer, shap_values, df_x_test, list_patients, filename_pattern):

    shap.decision_plot(shap_explainer.expected_value,
                       shap_values[list_patients, :],
                       feature_names=df_x_test.columns.values,
                       highlight=[0, 1, 2, 3, 4],
                       show=False,
                       )

    plt.savefig(
        str(Path.joinpath(consts.PATH_PROJECT_INTERPRETER, 'shap_decision_plot_{}.png'.format(filename_pattern))),
        bbox_inches='tight'
    )
    plt.close()


def plot_summary_plot(shap_values, df_x_test, filename_pattern):
    shap.summary_plot(shap_values, df_x_test, plot_type="bar", show=False)
    plt.savefig(Path.joinpath(consts.PATH_PROJECT_INTERPRETER, 'shap_summary_plot_{}.png'.format(filename_pattern)))
    plt.close()


def plot_summary_plot_average(dict_params, filename_pattern, flag_save_figure=True):
    path_shap_feature_importances = Path.joinpath(consts.PATH_PROJECT_METRICS, 'shap_importances.csv')
    df_shap_importances = pd.read_csv(str(path_shap_feature_importances))

    df_shap_importances = df_shap_importances.rename(columns={
        'album_0': 'Normoalbuminuria',
        'album_1': 'Microalbuminuria',
        'album_2': 'Macroalbuminuria',
        'ldl': 'LDL',
        'egfr': 'EGFR',
        'sbp': 'SBP',
        'dm_duration': 'DM Duration',
        'age': 'Age',
        'sex': 'Sex',
        'hba1c': 'Hba1c',
        'exercise': 'Exercise',
        'smoking': 'Smoking'
    })

    qry = ' and '.join(["({}=='{}')".format(k, v) for k, v in dict_params.items()])
    df_shap_importances_filter = df_shap_importances.query(qry)

    df_shap_importances_filter = df_shap_importances_filter.drop(columns=['seed', 'fes', 'type_over'])
    df_avg = df_shap_importances_filter.groupby(["estimator"]).agg([np.mean, np.std])
    df_avg.columns = df_avg.columns.map("_".join)
    mean_ids = ['{}_{}'.format(col_name, 'mean') for col_name in df_shap_importances_filter.columns.to_list()]
    std_ids = ['{}_{}'.format(col_name, 'std') for col_name in df_shap_importances_filter.columns.to_list()]
    mean_ids.remove('estimator_mean')
    std_ids.remove('estimator_std')

    df_mean = df_avg.T.loc[mean_ids].stack().reset_index().rename(columns={'level_0': 'var_name', 0: 'mean'})
    df_std = df_avg.T.loc[std_ids].stack().reset_index().rename(columns={'level_0': 'var_name', 0: 'std'})

    df_mean['idx'] = df_mean.apply(lambda x: '{}_{}'.format(x.var_name.split('_')[0], x.estimator), axis=1)
    df_std['idx'] = df_std.apply(lambda x: '{}_{}'.format(x.var_name.split('_')[0], x.estimator), axis=1)
    df_merge = pd.merge(df_mean, df_std, how='inner', on='idx')
    df_merge['var_name'] = df_merge['var_name_x'].apply(lambda x: x.split('_')[0])
    df_merge['estimator'] = df_merge['estimator_x'].apply(lambda x: x.upper())
    df_merge = df_merge.drop(columns=['estimator_x', 'estimator_y', 'var_name_x', 'var_name_y'])

    dfx = df_merge.pivot(index='var_name', columns='estimator').fillna(0).stack().reset_index()

    colors = ['green', 'red', 'blue', 'black']
    positions = [-1, 0, 1, 2]

    fig, ax = plt.subplots(figsize=(8, 12))
    for group, color, pos in zip(dfx.groupby('estimator'), colors, positions):
        key, group = group

        group.sort_values(['mean'], inplace=True)

        group.plot('var_name', 'mean', xerr='std', kind='barh',
                                             width=0.22, label=key, edgecolor='white', linewidth=2,
                                             position=pos, color=color, alpha=0.7, ax=ax)

    ax.set_ylim(-1, 12)
    plt.xlabel('SHAP value', fontsize=16)
    plt.ylabel('')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(axis='both', color='lightgray', alpha=0.5, linewidth=1.5, linestyle='--')
    plt.tight_layout()

    if flag_save_figure:
        plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, 'shap_avg_{}.pdf'.format(filename_pattern))))
        plt.close()
    else:
        plt.show()


parser = argparse.ArgumentParser(description='Process representations.')
args = parse_arguments(parser)

for idx in np.arange(1, 2, 1):

    filename_pattern = '{}_{}_{}_{}'.format(args.fes, args.type_over, args.estimator, idx)

    dict_params = {
        'fes': args.fes,
        'type_over': args.type_over,
    }

    seed = np.random.seed(idx)
    rng = np.random.RandomState(idx)

    df_x_train, y_train, df_x_test, y_test, df_x_train_raw, df_x_test_raw, dict_list_patients = load_train_test_partitions(filename_pattern, seed_value=idx)

    plot_summary_plot_average(dict_params, filename_pattern)

    shap_explainer = load_shap_model(filename_pattern)
    shap_values = shap_explainer.shap_values(df_x_test)

    plot_summary_plot(shap_values, df_x_test, filename_pattern)

    if args.type_cvd == 'low':
        print('cvd low risk')
        list_patients = dict_list_patients['cvd_low']
    elif args.type_cvd == 'moderate':
        print('cvd moderate risk')
        list_patients = dict_list_patients['cvd_moderate']
    elif args.type_cvd == 'high':
        print('cvd high risk')
        list_patients = dict_list_patients['cvd_high']
    else:
        list_patients = dict_list_patients['cvd_low'][10:15] + dict_list_patients['cvd_high'][:5]

    plot_decision_plot(shap_explainer, shap_values, df_x_test, list_patients, filename_pattern)

    df_x_test_raw = df_x_test_raw.round(2)
    df_x_test_raw['Age'] = df_x_test_raw['Age'].astype(int)
    df_x_test_raw['Diabetes duration'] = df_x_test_raw['Diabetes duration'].astype(int)

    # for id_patient in list_patients:
    #     plot_waterfall(shap_explainer, df_x_test, df_x_test_raw, y_test, filename_pattern, id_patient)
    #
    for id_patient in list_patients:
        plot_forceplot(shap_explainer, shap_values, df_x_test_raw, y_test, filename_pattern, id_patient)
