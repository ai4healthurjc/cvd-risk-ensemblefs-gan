import numpy as np
from pathlib import Path
from keras import layers
from keras.models import Sequential
from keras.layers import Activation
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
from keras import metrics
import matplotlib.pyplot as plt
from typing import Dict, Iterable, Any
import utils.consts as consts
import tensorflow as tf
import keras.backend as K
import random as python_random


def compute_mrae(y_real, y_pred):
    epison = 1e-6
    difference = K.abs(y_real - y_pred) / (y_real + epison)
    return K.mean(difference)


class MLPKerasRegressor(KerasRegressor):

    def __init__(
        self,
        hidden_layer_sizes=(100, ),
        activation="relu",
        kernel_init="random_uniform",
        optimizer="adam",
        epochs=200,
        verbose=0,
        n_features_in=12,
        seed_value=3232,
        flag_save_figure=False,
        learning_rate=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose
        self.activation = activation
        self.kernel_init = kernel_init
        self.n_features_in_ = n_features_in
        self.learning_rate = learning_rate
        self.flag_save_figure = flag_save_figure

        np.random.seed(seed_value)
        python_random.seed(seed_value)
        tf.random.set_seed(seed_value)

    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        model = Sequential()

        for i, nodes in enumerate(self.hidden_layer_sizes):
            if i == 0:
                model.add(layers.Dense(nodes, input_dim=self.n_features_in_, kernel_initializer=self.kernel_init, activation=self.activation))
                # model.add(Activation(self.activation))
                # model.add(Activation(self.activation))
            else:
                model.add(layers.Dense(nodes, kernel_initializer=self.kernel_init, activation=self.activation))
                # model.add(layers.Dropout(0.1))
                # model.add(Activation(self.activation))

        model.add(layers.Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss="mean_squared_error",
                      optimizer=compile_kwargs["optimizer"],
                      metrics=[metrics.mean_squared_error,
                               metrics.mean_absolute_error,
                               compute_mrae])

        return model


class MLPRegressor(object):

    def __init__(self, layers, activation, loss_func, loss_metric, optimizer, init, input_dim, patience, verbose, flag_save_figure):
        self.model = Sequential()
        self.layers = layers
        self.activation = activation
        self.loss_func = loss_func
        self.loss_metric = loss_metric
        self.opt = optimizer
        self.init = init
        self.input_dim = input_dim
        self.patience = patience
        self.verbose = verbose
        self.flag_save_figure = flag_save_figure
        self.seed_value = 0
        self._create_model()

    def _create_model(self):

        for i, nodes in enumerate(self.layers):
            if i == 0:
                self.model.add(layers.Dense(nodes, input_dim=self.input_dim, kernel_initializer=self.init))
                self.model.add(Activation(self.activation))
            else:
                self.model.add(layers.Dense(nodes, kernel_initializer=self.init))
                self.model.add(Activation(self.activation))

        self.model.add(layers.Dense(1))
        self.model.add(Activation('sigmoid'))
        # self.model.add(Activation('linear'))

        print(self.model.summary())

        self.model.compile(optimizer=self.opt, loss=self.loss_func, metrics=[self.loss_metric])

    def fit(self, x_train, y_train, x_validation, y_validation, batch_size, epochs, random_state):

        self.seed_value = random_state

        early_stopping = EarlyStopping(monitor='val_{}'.format(self.loss_metric), mode='min', verbose=self.verbose, patience=self.patience,
                                       restore_best_weights=True)
        callbacks_list = [early_stopping]

        np.random.seed(random_state)
        history = self.model.fit(x_train, y_train, validation_data=(x_validation, y_validation),
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=self.verbose,
                                 callbacks=callbacks_list)

        self._plot_learning_curve(history)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def _plot_learning_curve(self, history):

        fig, ax = plt.subplots()

        plt.plot(history.history[self.loss_metric])
        plt.plot(history.history['val_{}'.format(self.loss_metric)])
        plt.xlabel('epochs')
        plt.ylabel(self.loss_metric)

        if self.flag_save_figure:
            fig.tight_layout()
            fig.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, 'history_{}.pdf'.format(self.seed_value))))
        else:
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()


