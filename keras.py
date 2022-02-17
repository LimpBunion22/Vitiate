import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import io
from contextlib import redirect_stdout



def create_keras_net(inputs_n, layers_n, neurons_per_layer):

    l_list = []

    for l in range(layers_n):
        l_list.append(layers.Dense(
            neurons_per_layer[l], activation="relu", name="layer"+str(l)))

    keras_model = keras.Sequential(l_list)
    keras_model(tf.ones((inputs_n, 1)))

    return keras_model


def create_test_input(inputs_n):

    return tf.ones((int(inputs_n), 1))


def run_backward(keras_model, structure, iterations):

    inputs = tf.keras.layers.Input(shape=(structure['inputs_n'],))
    outputs = tf.keras.layers.Dense(1)(inputs)
    keras_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    keras_model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.MeanAbsoluteError(),
        # List of metrics to monitor
        metrics=[keras.metrics.MeanAbsoluteError()])

    x = np.ndarray(shape=(1, structure['inputs_n']), dtype=float,
                   order='F', buffer=np.ones(structure['inputs_n']))
    y = 5*np.ndarray(shape=(1, int(structure['neurons_per_layer'][structure['layers_n']-1])), dtype=float,
                   order='F', buffer=np.ones(int(structure['neurons_per_layer'][structure['layers_n']-1])))
    data_set = tf.data.Dataset.from_tensors((x, y)).repeat(iterations)

    f = io.StringIO()
    with redirect_stdout(f):
        keras_model.fit(
            data_set,
            batch_size=1,
            steps_per_epoch=iterations)
    # out = f.getvalue()
    

    return
