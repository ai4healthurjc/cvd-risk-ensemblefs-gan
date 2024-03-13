import numpy as np
import pandas as pd
import shap
import math
import joblib
# import seaborn as sns
from pathlib import Path
# from PyALE import ale
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_sequential_feature_selection
# from yellowbrick.model_selection import ValidationCurve
# import matplotlib.patches as patches
# from sklearn.model_selection import validation_curve
import datetime
import itertools
from utils.metrics import compute_mrae, compute_mae
from utils.loader import load_train_test_partitions, load_model_estimator
import utils.consts as consts


def plot_confusion_matrix(cm, class_names, flag_save_figure, cmap=plt.cm.Blues):
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks, rotation=0)
    ax = plt.gca()
    # ax.set_xticklabels((ax.get_xticks() + 1).astype(str))
    plt.yticks(tick_marks)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.1f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if flag_save_figure:
        plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, '{}_confusion_matrix.png'.format(fs_method))))
        plt.close()
    else:
        plt.show()


def plot_summary_survey(results, category_names):
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (col_name, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5, label=col_name, color=color)
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)

    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')
    # ax.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.show()


def plot_stacked_barplot(df, example=False):
    # Define function to process input data into rectangle left corner indices
    def process_data(array):
        left = np.zeros_like(array)
        mid = array[3] / 2
        left[0] = -np.sum(array[0:3]) - mid
        left[1] = -np.sum(array[1:3]) - mid
        left[2] = -np.sum(array[2:3]) - mid
        left[3] = -mid
        left[4] = mid
        left[5] = np.sum(array[4:5]) + mid
        left[6] = np.sum(array[4:6]) + mid
        width = array
        return left, width

    def plot_rect(bottom, left, width, height, color='C0'):
        ax.add_patch(patches.Rectangle((left, bottom), width, height, linewidth=1, edgecolor=color, facecolor=color))

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Define axis ticks ticks
    plt.xticks(np.arange(-1, 1.25, 0.25), np.arange(-100, 125, 25))
    plt.yticks(np.arange(0, 1.2, 0.2), np.arange(0, 1.2, 0.2))

    # Define axis limits
    plt.ylim(0.05, 0.95)
    plt.xlim(-1.125, 1.125)

    # Move gridlines to the back of the plot
    plt.gca().set_axisbelow(True)

    # Change color of plot edges
    ax.spines['left'].set_color('lightgray')
    ax.spines['right'].set_color('lightgray')
    ax.spines['top'].set_color('lightgray')

    # Hide y axis ticks
    plt.gca().tick_params(axis='y', colors='w')

    # Turn on gridlines and set color
    plt.grid(b=True, axis='both', color='lightgray', alpha=0.5, linewidth=1.5)

    # Add lines
    plt.axvline(x=0, c='lightgray')
    plt.axhline(y=0.5, c='black')

    # Add x label
    plt.xlabel('Percent', fontsize=14)

    # Define color scheme from negative to positive
    colors = ['firebrick', 'sandybrown', 'navajowhite',
              'khaki', 'lightcyan', 'skyblue', 'steelblue']

    # Process data to plot
    try:
        array = [df.iloc[0, :].values,
                 df.iloc[1, :].values,
                 df.iloc[2, :].values,
                 df.iloc[3, :].values]
    except:
        print('Plotting example data')
        example = True

    if example == True:
        # Example data
        array = [np.array([0.05, 0.1, 0.2, 0.2, 0.3, 0.1, 0.05]),
                 np.array([0, 0.1, 0.1, 0.3, 0.2, 0.2, 0.1]),
                 np.array([0.1, 0.2, 0.2, 0.3, 0.1, 0.05, 0.05]),
                 np.array([0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1])]

        # Example data column names
        df = pd.DataFrame(columns=['Completely Dissatisfied',
                                   'Somewhat Dissatisfied',
                                   'Dissatisfied',
                                   'Neither Satisfied nor Dissatisfied',
                                   'Somewhat Satisfied',
                                   'Satisfied',
                                   'Completely Satisfied'])
    # Compute average statistics
    hi = [sum(x[4:]) for x in array]
    med = [x[3] for x in array]
    lo = [sum(x[0:3]) for x in array]

    left = {}
    width = {}
    for i in range(4):
        left[i], width[i] = process_data(array[i])

    # Plot boxes
    height = 0.13
    bottom = 0.135
    for i in range(len(array)):
        for j in range(len(array[i])):
            plot_rect(bottom=bottom + i * 0.2, left=left[i][j], width=width[i][j], height=height, color=colors[j])

    # Plot category labels
    plt.text(-1.1, 0.9, 'Unfavorable', style='italic',
             horizontalalignment='left', verticalalignment='center')
    plt.text(0, 0.9, 'Neutral', style='italic',
             horizontalalignment='center', verticalalignment='center')
    plt.text(1.1, 0.9, 'Favorable', style='italic',
             horizontalalignment='right', verticalalignment='center')

    # Plot percentages
    for i in range(len(med)):
        plt.text(-1, 0.2 * (i + 1), '{0:.0%}'.format(lo[i]),
                 horizontalalignment='left', verticalalignment='center')
        plt.text(0, 0.2 * (i + 1), '{0:.0%}'.format(med[i]),
                 horizontalalignment='center', verticalalignment='center')
        plt.text(1, 0.2 * (i + 1), '{0:.0%}'.format(hi[i]),
                 horizontalalignment='right', verticalalignment='center')

    # Create legend
    fig, ax = plt.subplots(1, figsize=(6, 2))
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot colored circles
    legend_left = [-0.9, -0.6, -0.3, 0, 0.30, 0.6, 0.9]
    for i in range(len(colors)):
        plot_rect(bottom=0, left=legend_left[i], width=0.2, height=0.2, color=colors[i])

    # Plot labels 1-6
    for i in range(0, 6, 2):
        plt.text(-0.8 + 0.3 * i, 0.25, df.columns[i].replace(' ', '\n'),
                 horizontalalignment='center', verticalalignment='bottom')
        plt.text(-0.5 + 0.3 * i, -0.05, df.columns[i + 1].replace(' ', '\n'),
                 horizontalalignment='center', verticalalignment='top')

    # Plot last label
    plt.text(1, 0.25, df.columns[6].replace(' ', '\n'),
             horizontalalignment='center', verticalalignment='bottom')

    # Plot label title
    plt.text(-1, 0.1, 'Scale', fontsize=14,
             horizontalalignment='right', verticalalignment='center')

    plt.gca().autoscale(enable=True, axis='both', tight=None)

    plt.show()


def graficos_filter(var_imp: pd.DataFrame, description: str, flag_save_figure=False):
    mean_df = var_imp.mean()
    df_rep = pd.DataFrame(mean_df, columns=['Importancia'])
    df_rep = df_rep.sort_values(by='Importancia', ascending=False)

    var = df_rep.index.tolist()

    plt.bar(var, df_rep['Importancia'])
    plt.title('Importancia de las variables utilizando ' + description, fontsize=16)
    plt.xlabel('\n Variables', fontsize=14)
    plt.ylabel('Importance \n', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(rotation=70)
    plt.show()

    dif = []
    nombres_val = []
    for j in range(1, len(df_rep['Importancia'])):
        dif.append((df_rep['Importancia'][j - 1]) - (df_rep['Importancia'][j]))
        nombres_val.append(var[j - 1] + '-' + var[j])

    # Grafico de las diferencias
    plt.bar(range(0, len(df_rep) - 1), dif)
    plt.title('Diferencias entre importancias utilizando ' + description)
    plt.xlabel('\n Variables', fontsize=14)
    plt.ylabel('Diferencia \n', fontsize=14)
    plt.xticks(range(0, len(df_rep) - 1), nombres_val)
    plt.xticks(fontsize=10)
    plt.xticks(rotation=70)

    if flag_save_figure:
        print('saving fig')
    else:
        plt.show()


def plot_time_series(df_ts, x, y, ids, title, xlabel, ylabel):
    # plt.figure(figsize=(15, 4))

    print(df_ts[df_ts['patient_id'] == 2].head(30))
    print(df_ts[df_ts['patient_id'] == 4].head(30))

    df_ts.set_index('date').groupby('patient_id').eversense_mgdl.plot(style='.-')

    plt.axhline(y=70, color='red', linestyle='--')
    plt.text(0, 70 + 5, 'hypoglycemia')
    plt.axhline(y=180, color='black', linestyle='--')
    plt.text(0, 180 + 5, 'hyperglycemia')

    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_fs_score(df_scores, fs_method, k_features, estimator_name=None, type_over='wo', seed_value=3232, flag_save_figure=False):
    plt.figure(figsize=(6, 4))
    # fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    plt.bar(df_scores['var_name'].values, df_scores['score'].values)
    plt.xlabel('Feature names')
    plt.ylabel('Score')

    plt.xticks(rotation=90)
    plt.grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.title('{}'.format(fs_method))
    plt.legend()

    if flag_save_figure:
        plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, '{}_{}_k_{}_{}_seed_{}_score_importance.png'.format(estimator_name,
                                                                                                                    fs_method,
                                                                                                                    k_features,
                                                                                                                    type_over,
                                                                                                                    seed_value))))
        plt.close()
    else:
        plt.show()


def plot_mean_std_metric(df_metrics, lims, metric_name, flag_save_figure=False):
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

    ax.plot(df_metrics['Media'], df_metrics['Modelo'], ls='', marker='o', color='#8000FF')

    ax.hlines(df_metrics['Modelo'], df_metrics['Media'] - df_metrics['Std'], df_metrics['Media'] + df_metrics['Std'],
              label='', lw=2, color='#8000FF', ls='-')

    ax.grid(axis='x', ls='-')
    ax.grid(axis='y', ls=':', lw=1, alpha=0.5)
    ax.set(xlabel=metric_name, xlim=lims)

    fig.tight_layout()

    if flag_save_figure:
        fig.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES,
                                      '{}_loss_metric_{}_{}.png'.format(title_file, loss_metric, seed_value))))
        plt.close()
    else:
        plt.show()


def plot_sfs(sfs1, kind, title, flag_save_figure=False):
    plot_sequential_feature_selection(sfs1.get_metric_dict(), kind=kind)
    plt.title(title)
    plt.grid()

    if flag_save_figure:
        print('saving figure')
    else:
        plt.show()


def plot_learning_curve_mlp(history, title_file, seed_value, list_regression_metrics, flag_save_figure=False):
    for loss_metric in list_regression_metrics:

        fig, ax = plt.subplots()

        plt.plot(history[loss_metric])
        plt.plot(history['val_{}'.format(loss_metric)])
        plt.xlabel('epochs')
        plt.ylabel(loss_metric)
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()

        if flag_save_figure:
            fig.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES,
                                          '{}_loss_metric_{}_{}.png'.format(title_file, loss_metric, seed_value))))
            plt.close()
        else:
            plt.show()


# def plot_learning_curve_xxx(x_features, y_label, test_size_proportion, flag_save_figure):
#
#     for idx in np.arange(1, 6, 1):
#         x_train, x_test, y_train, y_test = train_test_split(x_features, y_label,
#                                                             test_size=test_size_proportion,
#                                                             random_state=idx)
#
#         train_score, test_score = validation_curve(dict_params_gridsearch['regressor'], x_train, y_train,
#                                                    param_name=dict_params_gridsearch['param_name'],
#                                                    param_range=dict_params_gridsearch['param_range'],
#                                                    cv=dict_params_gridsearch['cv'], scoring=dict_params_gridsearch['score'])
#
#         mean_train_score = np.mean(train_score, axis=1)
#         std_train_score = np.std(train_score, axis=1)
#
#         mean_test_score = np.mean(test_score, axis=1)
#         std_test_score = np.std(test_score, axis=1)
#
#         fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
#         ax.plot(dict_params_gridsearch['param_range'], mean_train_score, label="Training Score", color='b')
#         ax.plot(dict_params_gridsearch['param_range'], mean_test_score, label="Validation Score", color='g')
#
#         ax.set_xlabel(dict_params_gridsearch['param_name'])
#         ax.set_ylabel(dict_params_gridsearch['score'])
#         plt.tight_layout()
#         plt.legend(loc='best')
#
#         if flag_save_figure:
#             fig.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, 'learning_curve_partition_{}.pdf'.format(idx))))
#         else:
#             plt.show()
#
#     fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
#
#     viz = ValidationCurve(
#         dict_params_gridsearch['regressor'], dict_params_gridsearch['param_name'],
#         param_range=dict_params_gridsearch['param_range'], cv=dict_params_gridsearch['cv'], scoring=dict_params_gridsearch['score']
#     )
#
#     viz.fit(x_features, y_label)
#     if flag_save_figure:
#         fig.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, 'learning_curve_partition_average.pdf')))
#     else:
#         viz.show()


def plot_summary_plot(model, x_train, x_test, v_column_names):
    explainer = shap.DeepExplainer(model, x_train)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values[0], plot_type='bar', feature_names=v_column_names)


def plot_learning_curve_hyperparameter(x_train, y_train, regressor_name, model_regressor, seed_value,
                                       param_name, param_range, cv, scoring, grid_best_params,
                                       flag_save_figure=False):
    viz = ValidationCurve(
        estimator=model_regressor, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring,
    )

    viz.fit(x_train, y_train)
    plt.xlabel(param_name)
    plt.ylabel(scoring)
    plt.suptitle(str(grid_best_params), fontsize=12)

    if flag_save_figure:
        out_path = str(Path.joinpath(consts.PATH_PROJECT_FIGURES,
                                     '{}_learning_curve_average_{}_{}.png'.format(regressor_name, param_name,
                                                                                  seed_value)))

    else:
        out_path = None

    viz.show(outpath=out_path)
    plt.close()


def plot_corresponding_ale_categorical_feature(df_ale_cat, var_name,
                                               estimator_name, fes,
                                               type_over, seed_value, flag_save_figure=True):

    df_ale_cat = df_ale_cat.reset_index()
    first_std = df_ale_cat.iloc[0, 1]
    df_ale_cat['lowerCI_95%'] = df_ale_cat['lowerCI_95%'].fillna(first_std)
    df_ale_cat['upperCI_95%'] = df_ale_cat['upperCI_95%'].fillna(first_std)

    save_ci_area_csv(df_ale_cat, var_name, fes, type_over, estimator_name, seed_value)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.1))

    ax.bar(df_ale_cat[var_name], df_ale_cat['eff'], width=0.4, color='blue', alpha=0.0)
    ax.set_ylabel('', fontsize=16)

    ax2 = ax.twinx()
    ax2.bar(df_ale_cat[var_name], df_ale_cat['size'], width=0.4, color='blue', alpha=0.3)
    ax2.set_ylabel('Size', fontsize=16)

    ax.plot([df_ale_cat.iloc[0, 0], df_ale_cat.iloc[1, 0]],
            [df_ale_cat.iloc[0, 1], df_ale_cat.iloc[1, 1]],
            '--', marker='o', color='black')

    for index, row in df_ale_cat.iterrows():
        p_inf = [row[var_name], row['lowerCI_95%']]
        p_center = [row[var_name], row['eff']]
        yerr = math.sqrt((p_center[0] - p_inf[0])**2 + (p_center[1] - p_inf[1])**2)

        ax.errorbar(row[var_name], row['eff'], yerr, fmt='o',
                    lw=2, capsize=4, color='black')

    ax.set_xlabel(var_name, fontsize=12)
    ax.set_xticks([0, 1])
    plt.grid(alpha=0.5, linestyle='--')
    fig.tight_layout()

    if flag_save_figure:
        fig.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, 'ale',
                                      '{}_{}_{}_ale_{}_seed_{}.png'.format(estimator_name,
                                                                           type_over,
                                                                           fes,
                                                                           var_name,
                                                                           seed_value))))
        plt.close()
    else:
        plt.show()


def plot_area_cis(estimator_name, type_over, fes, seed_value, flag_save_figure=False):
    df_ci = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_METRICS, 'area_ci.csv')))

    df_ci_filtered = df_ci[(df_ci['type_over'] == type_over) & (df_ci['fs'] == fes) & (df_ci['seed'] == seed_value)]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.bar(df_ci_filtered['var_name'], df_ci_filtered['area_ci'], width=0.4, color='blue')
    ax.set_xlabel('features', fontsize=12)
    ax.set_ylabel('CI', fontsize=12)
    ax.set_title('fs: {} and {}'.format(fes, type_over))

    if flag_save_figure:
        fig.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, 'ale',
                                      '{}_{}_{}_area_ci_seed_{}.png'.format(estimator_name,
                                                                        type_over,
                                                                        fes,
                                                                        seed_value))))
        plt.close()
    else:
        plt.show()


def save_ci_area_csv(df_ale_var, var_name, fes, type_over, estimator_name, seed_value):

    list_area_cis = []
    for index, row in df_ale_var.iterrows():
        area_ci = abs(row['upperCI_95%'] - row['eff']) * 2
        list_area_cis.append(area_ci)

    dict_area_ci = {
        'date': datetime.datetime.now(),
        'seed': seed_value,
        'fs': fes,
        'estimator': estimator_name,
        'type_over': type_over,
        'var_name': var_name,
        'area_ci': np.sum(np.array(list_area_cis))
    }

    df_ci = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_METRICS, 'area_ci.csv')))
    df_ci = df_ci.append(dict_area_ci, ignore_index=True)
    df_ci.to_csv(str(Path.joinpath(consts.PATH_PROJECT_METRICS, 'area_ci.csv')), index=False)


def plot_corresponding_ale_numerical_feature(df_ale_numerical, df_ale_numerical_raw, var_name,
                                             estimator_name, fes, type_over, seed_value, flag_save_figure=True):

    df_ale_numerical = df_ale_numerical.reset_index()
    df_ale_numerical_raw = df_ale_numerical_raw.reset_index()
    first_std = df_ale_numerical.iloc[0, 1]
    df_ale_numerical['lowerCI_95%'] = df_ale_numerical['lowerCI_95%'].fillna(first_std)
    df_ale_numerical['upperCI_95%'] = df_ale_numerical['upperCI_95%'].fillna(first_std)

    save_ci_area_csv(df_ale_numerical, var_name, fes, type_over, estimator_name, seed_value)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.1))

    ax.plot(df_ale_numerical_raw[var_name], df_ale_numerical['eff'], ls='solid', color='blue')
    ax.fill_between(df_ale_numerical_raw[var_name], df_ale_numerical['lowerCI_95%'],
                    df_ale_numerical['upperCI_95%'], label='95% CI',
                    alpha=0.3, color='grey', lw=2)

    ax.set_xlabel(var_name, fontsize=12)
    # ax.set_ylabel('Effect on prediction', fontsize=12)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend()
    plt.grid(alpha=0.5, linestyle='--')
    fig.tight_layout()

    if flag_save_figure:
        fig.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, 'ale',
                                      '{}_{}_{}_ale_{}_seed_{}.png'.format(estimator_name,
                                                                           type_over,
                                                                           fes,
                                                                           var_name,
                                                                           seed_value))))
        plt.close()
    else:
        plt.show()


def plot_scatter_from_model(fes, type_over, estimator, seed_value):

    filename_pattern = '{}_{}_{}_{}'.format(fes, type_over, estimator, seed_value)

    df_x_train_pre, y_train, df_x_test_pre, y_test, df_x_train_raw, df_x_test_raw, _ = load_train_test_partitions(filename_pattern, seed_value)
    estimator_model = load_model_estimator(fes, type_over, estimator, seed_value, df_x_train_pre.values, y_train)

    y_pred = estimator_model.predict(df_x_test_pre.values)

    plot_scatter_real_pred(y_test, y_pred,
                           title_figure='',
                           title_file='{}'.format(filename_pattern),
                           seed_value=seed_value,
                           flag_save_figure=True
                           )


def plot_shap_mean(fes, type_over, estimator, flag_save_figure=False):
    for seed_value in range(1, 6, 1):
        df_x_train_pre, y_train, df_x_test_pre, y_test, df_x_train_raw, df_x_test_raw, _ = load_train_test_partitions(seed_value)

        print(df_x_train_pre)

        path_shap_explainer = Path.joinpath(
            consts.PATH_PROJECT_INTERPRETER, 'shap_{}_{}_{}.bz2'.format(fes, estimator, seed_value)
        )

        shap_explainer = joblib.load(filename=str(path_shap_explainer))
        shap_values = shap_explainer.shap_values(df_x_test_pre)

        shap.summary_plot(shap_values, df_x_train_pre, max_display=10, show=False)
        plt.show()

    if flag_save_figure:
        plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, 'shap', 'shap_mean_{}_{}.pdf'.format(fes, type_over))),
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()



def plot_heatmap_selected_features_fes(fes, type_over, flag_save_figure=True):
    df_selected_features = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_METRICS, 'selected_features.csv')))
    df_selected_features = df_selected_features.drop(columns=['date'])
    df_selected_features = df_selected_features[(df_selected_features['fs_method'] == fes)
                                                & (df_selected_features['type_over'] == type_over)]

    df_gk = df_selected_features.groupby(['fs_method', 'type_over', 'estimator'], as_index=False).agg(['sum']).reset_index()
    df_gk.columns = df_gk.columns.droplevel(-1)

    names_estimators = df_gk['estimator'].values
    names_estimators = list(map(lambda x: x.upper(), names_estimators))
    names_vars = df_gk.iloc[:, 3:-1].columns.values
    m_heatmap = df_gk.iloc[:, 3:-1].values.T
    df_heatmap = pd.DataFrame(m_heatmap, columns=names_estimators, index=names_vars)
    df_heatmap = df_heatmap.rename(index={'age': 'Age',
                                          'album_0': 'Normoalbuminuria',
                                          'album_1': 'Microalbuminuria',
                                          'album_2': 'Macroalbuminuria',
                                          'dm_duration': 'DM Duration',
                                          'egfr': 'EGFR',
                                          'hba1c': 'Hba1c',
                                          'exercise': 'Exercise',
                                          'ldl': 'LDL',
                                          'sbp': 'SBP',
                                          'sex': 'Sex',
                                          'smoking': 'Smoking'
                                        }
                                    )
    df_heatmap = df_heatmap.sort_index(ascending=True)

    # fig, ax = plt.subplots(1, 3, figsize=(8, 6))
    im, _ = plot_heatmap(df_heatmap.values, df_heatmap.index.values, df_heatmap.columns.values,
                      vmin=0, vmax=5, cmap="magma_r", cbarlabel="Frequency")
    texts = annotate_heatmap(im, valfmt="{x:d}", size=11)
    # annotate_heatmap(im, valfmt="{x:d}", size=11, threshold=20)

    if flag_save_figure:
        plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, 'fs', 'heatmap_{}_{}.pdf'.format(fes, type_over))),
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_area_cis_average(fes, estimator, flag_save_figure):

    df_cis = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_METRICS, 'area_ci.csv')))
    df_ci_filtered = df_cis[(df_cis['fs'] == fes) & (df_cis['estimator'] == estimator)]

    df_gk = df_ci_filtered.groupby(['type_over', 'var_name'], as_index=False)['area_ci'].agg(['mean', 'std']).reset_index()
    df_gk['std'] = df_gk['std'].fillna(0)
    df_gk['std'] = df_gk['std'].values / 2

    path_table = str(Path.joinpath(consts.PATH_PROJECT_REPORTS, 'latex', 'table_{}_{}.tex'.format(estimator, fes)))

    with open(path_table, 'w') as tf:
        tf.write(df_gk.to_latex(caption='{}+{}'.format(fes, estimator)))

    ax = sns.barplot(data=df_gk, x='var_name', y='mean', hue='type_over')

    # for container in ax.containers:
    #     ax.bar_label(container, rotation=90)

    num_hues = len(np.unique(df_gk['type_over']))
    for (hue, df_hue), dogde_dist in zip(df_gk.groupby('type_over'), np.linspace(-0.4, 0.4, 2 * num_hues + 1)[1::2]):
        print(df_hue)
        bars = ax.errorbar(data=df_hue, x='var_name', y='mean', yerr='std', ls='', lw=3, color='black')
        xys = bars.lines[0].get_xydata()
        bars.remove()
        ax.errorbar(data=df_hue, x=xys[:, 0] + dogde_dist, y='mean', yerr='std', ls='', lw=3, color='black')

    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

    plt.xlabel('Feature')
    plt.ylabel('CI area')
    plt.grid(alpha=0.5, linestyle='--')
    plt.tight_layout()

    if flag_save_figure:
        plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, 'ale', '{}_{}_ale_plots_average.png'.format(estimator, fes))))
        plt.close()
    else:
        plt.show()


def plot_ale_features(df_x_test, v_removed_features, model_estimator, df_x_test_raw,
                      list_vars_numerical, list_vars_categorical, estimator_name, fes, type_over, seed_value):

    list_vars_numerical = list(set(list_vars_numerical) - set(list(v_removed_features)))
    list_vars_categorical = list(set(list_vars_categorical) - set(list(v_removed_features)))

    for var_numerical in list_vars_numerical:
        df_ale_numerical = ale(
            X=df_x_test,
            model=model_estimator,
            feature=[var_numerical],
            plot=False,
            feature_type="continuous",
            grid_size=10,
            include_CI=True,
            C=0.95,
        )

        df_ale_numerical_raw = ale(
            X=df_x_test_raw,
            model=model_estimator,
            feature=[var_numerical],
            plot=False,
            feature_type="continuous",
            grid_size=10,
            include_CI=True,
            C=0.95,
        )

        plot_corresponding_ale_numerical_feature(df_ale_numerical, df_ale_numerical_raw, var_numerical,
                                                 estimator_name, fes, type_over, seed_value)

    for var_categorical in list_vars_categorical:

        ale_discr = ale(
            X=df_x_test,
            model=model_estimator,
            feature=[var_categorical],
            feature_type="discrete",
            plot=False,
            grid_size=10,
            include_CI=True,
            C=0.95,
        )

        plot_corresponding_ale_categorical_feature(ale_discr, var_categorical,
                                                   estimator_name, fes, type_over, seed_value)


def plot_learning_curves_several_hyperparameters(x_train: np.array,
                                                 y_train: np.array,
                                                 regressor_name: str,
                                                 model_regressor,
                                                 seed_value: int,
                                                 list_dict_params: list,
                                                 cv: int,
                                                 scoring,
                                                 grid_best_params: dict,
                                                 flag_save_figure: False
                                                 ):

    for dict_specficic_hypeparam in list_dict_params:
        # plot_learning_curve_hyperparameter(x_train, y_train, regressor_name, model_regressor, seed_value,
        #                                    dict_specficic_hypeparam['param_name'],
        #                                    dict_specficic_hypeparam['param_range'],
        #                                    cv, scoring, grid_best_params, flag_save_figure
        #                                    )

        plot_learning_curve_partition(x_train, y_train, regressor_name, model_regressor, seed_value,
                                      dict_specficic_hypeparam['param_name'],
                                      dict_specficic_hypeparam['param_range'],
                                      cv, scoring, grid_best_params, flag_save_figure
                                      )


def plot_learning_curve_partition(x_train, y_train, regressor_name, model_regressor, seed_value,
                                  param_name, param_range, cv, scoring, grid_best_params,
                                  flag_save_figure=False):
    train_score, test_score = validation_curve(model_regressor, x_train, y_train,
                                               param_name=param_name,
                                               param_range=param_range,
                                               cv=cv,
                                               scoring=scoring
                                               )

    mean_train_score = np.mean(train_score, axis=1)
    std_train_score = np.std(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)
    std_test_score = np.std(test_score, axis=1)

    if regressor_name == 'mlp':
        param_range_custom = ['param-{}'.format(x_item) for x_item in param_range]
    else:
        param_range_custom = param_range

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.plot(param_range_custom, mean_train_score, '--o', label="Training score", color="blue", lw=2)
    ax.fill_between(param_range_custom, mean_train_score - std_train_score, mean_train_score + std_train_score,
                    alpha=0.2, color="blue", lw=2)

    ax.plot(param_range_custom, mean_test_score, '--o', label="Validation score", color="green", lw=2)
    ax.fill_between(param_range_custom, mean_test_score - std_test_score, mean_test_score + std_test_score,
                    alpha=0.2, color="green", lw=2)

    ax.set_xlabel(param_name, fontsize=24)
    ax.set_ylabel(scoring, fontsize=24)
    plt.xticks(rotation=90)
    plt.suptitle(str(grid_best_params), fontsize=12)
    plt.grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.legend(loc='best')

    if flag_save_figure:
        fig.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES,
                                      '{}_learning_curve_partition_{}_{}.png'.format(regressor_name,
                                                                                     param_name,
                                                                                     seed_value))))
        plt.close()
    else:
        plt.show()


def plot_hists_comparison(y_real, y_pred, title_file, seed_value, flag_save_figure):
    n_bins = 50

    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax0, ax1 = axes.flatten()

    x_aux = np.c_[y_real, y_pred]
    colors = ['red', 'blue']

    ax0.hist(x_aux, bins=n_bins, histtype='step', label=['real', 'predicted'], color=colors)
    ax1.hist(x_aux, bins=n_bins, histtype='stepfilled', alpha=0.3, label=['real', 'predicted'], color=colors)

    plt.grid(axis='y', alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.legend()

    if flag_save_figure:
        fig.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, '{}_hists_{}.png'.format(title_file, seed_value))))
        plt.close()
    else:
        plt.show()


def plot_scatter_real_pred(y_real, y_pred, title_figure, title_file, seed_value, flag_save_figure):

    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    x = np.arange(0, 1.05, 0.05)
    y = x
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.scatter(y_real, y_pred, s=12, c='None', edgecolors='blue')
    ax.plot(x, y, color='r', linestyle='solid')
    ax.set_xlabel('CVD risk by ST1RE', fontsize=12.5)
    ax.set_ylabel('Predicted CVD risk', fontsize=12.5)
    # ax.set_title(title_figure)
    plt.grid(alpha=0.5, linestyle='--')

    # plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.axis('equal')

    # ax.set_xticks([0, 2, 4, 6], fontsize=20)
    # ax.set_xticklabels(['zero', 'two', 'four', 'six'])

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if flag_save_figure:
        fig.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, 'scatter_{}.pdf'.format(title_file, seed_value))),
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def pooled_var(stds):
    n = 5
    return np.sqrt(sum((n - 1) * (stds ** 2)) / len(stds) * (n - 1))


def plot_gridsearch_results(grid_cv, grid_params):
    print(grid_cv.cv_results_)

    df = pd.DataFrame(grid_cv.cv_results_)
    results = ['mean_test_score',
               'mean_train_score',
               'std_test_score',
               'std_train_score']

    lw = 2
    fig, axs = plt.subplots(1, len(grid_params), figsize=(5 * len(grid_params), 7))
    # axes[0].set_ylabel("Score", fontsize=25)

    for (param_name, param_range), ax in zip(grid_params.items(), axs.ravel()):
        grouped_df = df.groupby(f'param_{param_name}')[results] \
            .agg({'mean_train_score': 'mean',
                  'mean_test_score': 'mean',
                  'std_train_score': pooled_var,
                  'std_test_score': pooled_var})

        print(grouped_df, param_name, param_range)

        # previous_group = df.groupby(f'param_{param_name}')[results]
        # axes[idx].set_xlabel(param_name, fontsize=30)
        # axes[idx].set_ylim(0.0, 1.1)

        ax.plot(param_range, grouped_df['mean_train_score'], label="Training score",
                color="blue", lw=lw)
        # axes[idx].fill_between(param_range, grouped_df['mean_train_score'] - grouped_df['std_train_score'],
        #                        grouped_df['mean_train_score'] + grouped_df['std_train_score'], alpha=0.2,
        #                        color="darkorange", lw=lw)
        ax.plot(param_range, grouped_df['mean_test_score'], label="Cross-validation score",
                color="green", lw=lw)
        # axes[idx].fill_between(param_range, grouped_df['mean_test_score'] - grouped_df['std_test_score'],
        #                        grouped_df['mean_test_score'] + grouped_df['std_test_score'], alpha=0.2,
        #                        color="navy", lw=lw)

    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.suptitle('Validation curves', fontsize=40)
    # fig.legend(handles, labels, loc=8, ncol=2, fontsize=20)

    # fig.subplots_adjust(bottom=0.25, top=0.85)
    plt.show()


def plot_performance_evolution_k_features(df_metrics, metric_name, estimator_name, fs_method_name, type_over, flag_save_figure: bool = False):

    df_metrics_filtered = df_metrics[df_metrics['metric'] == metric_name]
    df_metrics_filtered = df_metrics_filtered.astype({'k_features': int})

    colors = plt.cm.Set1
    fig, ax = plt.subplots()

    # for i, (key, grp) in enumerate(df_metrics_filtered.groupby(['fs_method'])):
    for i, (key, grp) in enumerate(df_metrics_filtered.groupby(['estimator'])):
        ax = grp.plot(ax=ax, kind='line', x='k_features', y='mean', style='--o', c=colors(i), label=key)
        ax.fill_between(grp['k_features'],
                        grp['mean'] + grp['std'],
                        grp['mean'] - grp['std'], color=colors(i),
                        alpha=0.15, lw=2)

    ax.set_xlabel('# features', fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    # plt.title('Evolution with {}'.format(estimator_name))
    plt.grid(alpha=0.5, linestyle='--')
    plt.legend(loc='best')

    if flag_save_figure:
        fig.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES,
                                      '{}_{}_{}_evolution_k_features_{}.png'.format(estimator_name, fs_method_name,
                                                                                    type_over, metric_name))))
        plt.close()
    else:
        plt.show()


def plot_quality_scores(df_metrics):
    param_range = df_metrics['ir'].values

    colors = plt.cm.Set1

    fig, ax = plt.subplots()

    ax.fill_between(param_range, df_metrics['mean'] - df_metrics['std'],
                    df_metrics['mean'] + df_metrics['std'], alpha=0.25)

    ax.plot(param_range, df_metrics['mean'].values)

    plt.show()


def plot_heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", fontsize=11, **kwargs):

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=fontsize)

    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(fontsize)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize=fontsize)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize=fontsize)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), rotation=90)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(True)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_area_cis_average_v0(fes, estimator):

    df_cis = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_METRICS, 'area_ci.csv')))
    df_ci_filtered = df_cis[(df_cis['fs'] == fes) & (df_cis['estimator'] == estimator)]

    df_gk = df_ci_filtered.groupby(['type_over', 'var_name'], as_index=False)['area_ci'].agg(['mean', 'std']).reset_index()
    n_vars = np.unique(df_gk['var_name']).shape[0]

    ax = sns.barplot(data=df_gk, x='var_name', y='mean', hue='type_over', ci='std')
    ax.errorbar(data=df_gk, x='var_name', y='mean', yerr='std', ls='', lw=3, color='black')
    for container in ax.containers:
        ax.bar_label(container, rotation=90)

    plt.show()
