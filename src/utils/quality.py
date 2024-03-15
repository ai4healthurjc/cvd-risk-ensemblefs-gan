import numpy as np
import pandas as pd
import torch
from pathlib import Path
# from sdmetrics.reports.utils import get_column_plot
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from ctgan import CTGANSynthesizer, TVAESynthesizer
from sdmetrics.reports.single_table import QualityReport
from ctgan.synthesizers import CTGAN, TVAE
import matplotlib.pyplot as plt
import utils.consts as consts
from utils.selenium import download_steno_cvd_results

import logging
import coloredlogs


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def join_real_synthetic_data(x_real, x_synthetic, y_real, y_synthetic):
    x_train_resampled = np.concatenate((x_real, x_synthetic), axis=0)
    y_train_resampled = np.concatenate((y_real, y_synthetic))
    # y_train_resampled = np.concatenate((y_train, np.reshape(np.full((n_samples_class_min_syn, 1), id_label_min), -1)))

    return x_train_resampled, y_train_resampled,


def assign_label_class_patients(x):
    if x['label'] < 0.1:
        return 0
    elif 0.1 <= x['label'] < 0.2:
        return 1
    else:
        return 2


def get_data_min_maj_classes(x_train, y_train, v_col_names):

    df_x_train = pd.DataFrame(x_train, columns=v_col_names)
    df_x_train['label'] = y_train

    # Add y_train_class with low, moderate and high risk patients
    df_x_train['label_risk'] = df_x_train.apply(assign_label_class_patients, axis=1)
    y_train_risk = df_x_train['label_risk'].values

    v_classes, v_samples_classes = np.unique(y_train_risk, return_counts=True)

    id_label_maj = v_classes[np.argmax(v_samples_classes)]
    n_samples_maj = v_samples_classes[np.argmax(v_samples_classes)]

    v_ids_samples_min = np.delete(v_classes, [np.argmax(v_samples_classes)])
    v_num_samples_min = np.delete(v_samples_classes, [np.argmax(v_samples_classes)])
    print('kkkkkkkkkkkk: ', v_classes, v_samples_classes, v_ids_samples_min, v_num_samples_min)

    df_x_train_with_label = pd.DataFrame(x_train, columns=v_col_names)
    df_x_train_with_label['label'] = y_train_risk

    df_x_train_class_maj_with_label = df_x_train_with_label[df_x_train_with_label.loc[:, 'label'] == id_label_maj]
    df_x_train_class_maj = df_x_train_class_maj_with_label.iloc[:, :-1]
    y_train_class_maj = df_x_train_class_maj_with_label.loc[:, 'label']

    list_data_min_classes = []

    for id_class_min, n_samples_min in zip(v_ids_samples_min, v_num_samples_min):
        df_x_train_class_min_with_label = df_x_train_with_label[
            df_x_train_with_label.loc[:, 'label'] == id_class_min]
        df_x_train_class_min = df_x_train_class_min_with_label.iloc[:, :-1]
        y_train_class_min = df_x_train_class_min_with_label.loc[:, 'label']
        n_synthetic_samples_min = n_samples_maj - n_samples_min

        list_data_min_classes.append(
            (df_x_train_class_min, y_train_class_min, id_class_min, n_synthetic_samples_min))

    return df_x_train_class_maj, y_train_class_maj, id_label_maj, list_data_min_classes


def decode_ohe(m_ohe):
    m_decoded = np.argmax(m_ohe, axis=1)
    return m_decoded


def get_df_synthetic_with_albuminuria_binary_vars(df_x_train_syn_raw):
    v_unique_album_found, _ = np.unique(df_x_train_syn_raw.album.values, return_counts=True)
    df_album_binary = pd.get_dummies(df_x_train_syn_raw.album, prefix='album')

    for unique_album in v_unique_album_found:
        df_x_train_syn_raw['album_{}'.format(unique_album)] = df_album_binary['album_{}'.format(unique_album)]

    v_unique_album = np.array([0, 1, 2])

    if len(v_unique_album_found) < 3:
        v_diff = np.setdiff1d(v_unique_album, v_unique_album_found)
        for id_album in v_diff:
            df_x_train_syn_raw['album_{}'.format(id_album)] = 0

    df_x_train_class_min_syn = df_x_train_syn_raw.drop(
        columns=['Unnamed: 0', 'ldl_unit', 'hba1c_unit', 'previous_cvd', 'album', 'label'])

    return df_x_train_class_min_syn


def join_real_synthetic_samples(x_train, y_train, type_oversampling, seed_value):

    if type_oversampling == 'class':

        list_x_train_min = []
        list_y_train_min = []

        for id_label_min in [0, 2]:

            path_patients_class_min = str(Path.joinpath(
                consts.PATH_PROJECT_OVERSAMPLING, 'df_synthetic_{}_{}_seed_{}.csv'.format(type_oversampling,
                                                                                          id_label_min,
                                                                                          seed_value)))

            df_x_train_class_min_syn_raw = pd.read_csv(path_patients_class_min)
            y_train_class_min_syn = df_x_train_class_min_syn_raw['label'].values

            df_x_train_class_min_syn = get_df_synthetic_with_albuminuria_binary_vars(df_x_train_class_min_syn_raw)

            list_x_train_min.append(df_x_train_class_min_syn.values)
            list_y_train_min.append(y_train_class_min_syn)

        x_train_min_syn = np.vstack(list_x_train_min)
        y_train_min_syn = np.hstack(list_y_train_min)

        x_train_resampled = np.concatenate((x_train, x_train_min_syn), axis=0)
        y_train_resampled = np.concatenate((y_train, y_train_min_syn))

        return x_train_resampled, y_train_resampled

    elif type_oversampling == 'percentage':

        ir_percentage = 25
        path_patients_class_min = str(Path.joinpath(consts.PATH_PROJECT_OVERSAMPLING,
                                                    'df_synthetic_{}_{}_seed_{}.csv'.format(type_oversampling,
                                                                                            ir_percentage,
                                                                                            seed_value)))

        df_x_train_class_min_syn_raw = pd.read_csv(path_patients_class_min)
        y_train_min_syn = df_x_train_class_min_syn_raw['label'].values

        logger.info('Loading real patients with shape: {}'.format(x_train.shape))
        logger.info('Loading synthetic patients with shape: {}'.format(df_x_train_class_min_syn_raw.shape))

        df_x_train_min_syn = get_df_synthetic_with_albuminuria_binary_vars(df_x_train_class_min_syn_raw)
        x_train_min_syn = df_x_train_min_syn.values

        x_train_resampled = np.concatenate((x_train, x_train_min_syn), axis=0)
        y_train_resampled = np.concatenate((y_train, y_train_min_syn))

        logger.info('Resampled dataset with real and synthetic dataset: {}'.format(x_train_resampled.shape))

        return x_train_resampled, y_train_resampled

    else:
        return x_train, y_train


def load_synthetic_data_by_type_over(type_over, id_label_min, seed):

    path_x_train_syn_class_min = Path.joinpath(consts.PATH_PROJECT_OVERSAMPLING,
                                               'df_synthetic_{}_{}_seed_{}.csv'.format(type_over, id_label_min, seed)
                                               )

    return pd.read_csv(str(path_x_train_syn_class_min))


def load_target_synthetic_data(type_over, seed):
    path_y_train_syn = Path.joinpath(consts.PATH_PROJECT_OVERSAMPLING,
                                     'stenoRiskReport_{}_seed_{}.csv'.format(type_over, seed)
                                     )
    return pd.read_csv(str(path_y_train_syn))


def generate_synthetic_data_with_ctgan(df_x_train_real,
                                       v_col_names,
                                       type_over,
                                       n_synthetic_samples,
                                       n_epochs, batch_size,
                                       id_label,
                                       seed_value,
                                       save_synthetic_data,
                                       train_ctgan,
                                       verbose,
                                       flag_cuda
                                       ):

    path_df_synthetic = str(Path.joinpath(consts.PATH_PROJECT_OVERSAMPLING,
                                          'df_synthetic_{}_{}_seed_{}.csv'.format(type_over,
                                                                                  id_label,
                                                                                  seed_value)))

    if train_ctgan:
        # oversampler_model = CTGANSynthesizer(epochs=n_epochs, batch_size=batch_size, verbose=True)
        oversampler_model = CTGAN(epochs=n_epochs, batch_size=batch_size, verbose=verbose, cuda=flag_cuda)
        oversampler_model.fit(df_x_train_real, v_col_names)
        df_train_syn = oversampler_model.sample(n_synthetic_samples)

        df_train_syn['ldl_unit'] = 1
        df_train_syn['hba1c_unit'] = 1
        df_train_syn['previous_cvd'] = 0

        if save_synthetic_data:
            df_train_syn.to_csv(path_df_synthetic)

        # Get CVD risk from steno risk calculator
        df_steno_results = download_steno_cvd_results(df_train_syn, path_df_synthetic, type_over, id_label,
                                                      seed_value)
        df_train_syn['label'] = df_steno_results.loc[:, 'cvd_risk_10y']

        if save_synthetic_data:
            df_train_syn.to_csv(path_df_synthetic)

    else:  # load synthetic data
        df_train_syn = load_synthetic_data_by_type_over(type_over, id_label, seed_value)

    return df_train_syn


def generate_concatenate_real_synthetic_samples(x_train: np.array,
                                                y_train: np.array,
                                                v_col_names: np.array,
                                                seed_value,
                                                type_over='class',
                                                save_synthetic_data=False,
                                                train_ctgan=False,
                                                n_epochs=200,
                                                batch_size=40,
                                                flag_cuda=False
                                                ):

    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

    list_x_train_resampled = []
    list_y_train_resampled = []

    if type_over == 'class':

        df_x_train_class_maj, y_train_class_maj, id_label_maj, list_min_classes = get_data_min_maj_classes(x_train,
                                                                                                           y_train,
                                                                                                           v_col_names,
                                                                                                           )

        print('xxxxxxx')
        print(y_train_class_maj.shape)
        print('xxxxxxx')

        for df_x_train_class_min_real, y_train_class_min_real, id_label_min, n_synthetic_samples in list_min_classes:

            logger.info('seed: {}, x-class-min-real: {}, y-class-min-real: {}, label-class-min: {}, n-syn-samples: {}'
                        .format(seed_value,
                                df_x_train_class_min_real.shape,
                                y_train_class_min_real.shape,
                                id_label_min,
                                n_synthetic_samples
                                )
                        )

            df_train_class_min_syn = generate_synthetic_data_with_ctgan(df_x_train_class_min_real,
                                                                        v_col_names,
                                                                        type_over,
                                                                        n_synthetic_samples,
                                                                        n_epochs,
                                                                        batch_size,
                                                                        id_label_min,
                                                                        seed_value,
                                                                        train_ctgan=train_ctgan,
                                                                        save_synthetic_data=save_synthetic_data,
                                                                        verbose=True,
                                                                        flag_cuda=flag_cuda
                                                                        )

            x_train_resampled_class_min, y_train_resampled_class_min = join_real_synthetic_data(
                df_x_train_class_min_real.values,
                df_train_class_min_syn.loc[:, v_col_names].values,
                y_train_class_min_real,
                df_train_class_min_syn.loc[:, 'label'].values
                )

            list_x_train_resampled.append(x_train_resampled_class_min)
            list_y_train_resampled.append(y_train_resampled_class_min)

        x_train_min = np.vstack(list_x_train_resampled)
        y_train_min = np.hstack(list_y_train_resampled)

        x_train_resampled = np.concatenate((df_x_train_class_maj.values, x_train_min), axis=0)
        y_train_resampled = np.concatenate((y_train_class_maj, y_train_min))

    else:
        df_x_train_real = pd.DataFrame(x_train, columns=v_col_names)
        id_label = 25
        n_synthetic_samples = int(df_x_train_real.shape[0] * (id_label / 100.0))

        df_train_syn = generate_synthetic_data_with_ctgan(df_x_train_real,
                                                          v_col_names,
                                                          type_over,
                                                          n_synthetic_samples,
                                                          n_epochs,
                                                          batch_size,
                                                          id_label,
                                                          seed_value,
                                                          train_ctgan=train_ctgan,
                                                          save_synthetic_data=save_synthetic_data,
                                                          verbose=True,
                                                          flag_cuda=flag_cuda
                                                          )

        x_train_resampled, y_train_resampled = join_real_synthetic_data(df_x_train_real.values,
                                                                        df_train_syn.loc[:, v_col_names].values,
                                                                        y_train,
                                                                        df_train_syn.loc[:, 'label'].values
                                                                        )

    return x_train_resampled, y_train_resampled


def compute_quality_metrics(df_real, df_synthetic, list_categorical_vars, regressor_name,
                            id_class_min, seed_value, return_score=False, flag_save_figure=True):
    dict_aux = {}
    for var_name in df_real.columns.values:
        if var_name in list_categorical_vars:
            dict_aux[var_name] = {'type': 'categorical'}
        else:
            dict_aux[var_name] = {'type': 'numerical'}

    dict_metadata = {'fields': dict_aux}

    report = QualityReport()
    report.generate(df_real, df_synthetic, dict_metadata)

    df_metrics = report.get_details(property_name='Column Shapes')

    if return_score:
        qscore = df_metrics.iloc[:, -1].values
        return qscore
    else:
        fig = report.get_visualization(property_name='Column Shapes')

        if flag_save_figure:
            fig.write_image(str(Path.joinpath(consts.PATH_PROJECT_FIGURES,
                                              '{}_class_{}_univariate_similarity_{}.png'.format(regressor_name, id_class_min, seed_value))))
            plt.close()
        else:
            plt.show()


# df_x_train_synthetic_steno = df_x_train_synthetic.copy()
        # df_album = df_x_train_synthetic_steno.loc[:, ['album_0', 'album_1', 'album_2']]
        # m_album_decode = decode_ohe(df_album.values)
        # df_album['album'] = m_album_decode
        # print(df_album)
        # print(m_album_decode)


# list_qscores = []

# for current_ir in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
#     n_synthetic_samples = int(df_x_train_real.shape[0] * current_ir)
#
#     print('xxxxx: ', current_ir, df_x_train_real.shape[0], n_synthetic_samples)
#     oversampler_model = CTGANSynthesizer(epochs=n_epochs, batch_size=batch_size)
#     oversampler_model.fit(df_x_train_real, v_column_names)
#     df_x_train_synthetic = oversampler_model.sample(n_synthetic_samples)
#
#     qscore = compute_quality_metrics(df_x_train_real, df_x_train_synthetic,
#                                      list_categorical_vars, regressor_name, return_score=True,
#                                      id_class_min=1000, seed_value=seed_value)
#
#     print(seed_value, current_ir, np.mean(qscore))
#     list_qscores.append((seed_value, current_ir, np.mean(qscore)))
#
# return list_qscores


# compute_quality_metrics(df_x_train_class_min_real, df_x_train_class_min_syn,
            #                         list_categorical_vars, regressor_name, id_label_min, seed_value)

#
# def concatenate_real_synthetic_steno_data(x_train: np.array,
#                                           y_train: np.array,
#                                           v_col_names: np.array,
#                                           type_over: str,
#                                           seed: int,
#                                           ):
#
#     logger.info('Concatenating real and synthetic steno data')
#
#     if type_over == 'class':
#
#         df_x_train_class_maj, y_train_class_maj, id_label_maj, list_min_classes = get_data_min_maj_classes(x_train,
#                                                                                                            y_train,
#                                                                                                            v_col_names,
#                                                                                                            )
#         list_x_train_min = []
#         list_y_train_min = []
#
#         for df_x_train_class_min_real, y_train_class_min_real, id_label_min, n_synthetic_samples in list_min_classes:
#
#             logger.info('seed: {}, x-class-min-real: {}, y-class-min-real: {}, label-class-min: {}, n-syn-samples: {}'
#                         .format(seed,
#                                 df_x_train_class_min_real.shape,
#                                 y_train_class_min_real.shape,
#                                 id_label_min,
#                                 n_synthetic_samples
#                                 )
#                         )
#
#             df_x_train_class_min_syn = load_synthetic_data_by_type_over(type_over, id_label_min, seed)
#
#             x_train_resampled_class_min, y_train_resampled_class_min = join_real_synthetic_data(
#                 df_x_train_class_min_real.values,
#                 df_x_train_class_min_syn.values,
#                 y_train_class_min_real,
#                 n_synthetic_samples,
#                 id_label_min
#             )
#
#             list_x_train_min.append(x_train_resampled_class_min)
#             list_y_train_min.append(y_train_resampled_class_min)
#
#         x_train_min = np.vstack(list_x_train_min)
#         y_train_min = np.hstack(list_y_train_min)
#
#         x_train_resampled = np.concatenate((df_x_train_class_maj.values, x_train_min), axis=0)
#         y_train_resampled = np.concatenate((y_train_class_maj, y_train_min))
#
#     else: # percentage
#         df_x_train_real = pd.DataFrame(x_train, columns=v_col_names)
#         df_train_synthetic = load_synthetic_data_by_type_over(type_over, id_label_min=25, seed=seed)
#         df_x_train_synthetic = df_train_synthetic.loc[:, v_col_names]
#         y_train_synthetic = df_train_synthetic.loc[:, 'label']
#         x_train_resampled = np.concatenate((df_x_train_real.values, df_x_train_synthetic.values), axis=0)
#         y_train_resampled = np.concatenate((y_train, y_train_synthetic))
#
#     return x_train_resampled, y_train_resampled
#
