"""
================================================================================
DNN
--------------------------------------------------------------------------------
Implementation of the DNN baseline used in

T. O'Shea, J. Corgan, and T. C. Clancy,
"Convolutional Radio Modulation Recognition Networks,"
2016.

Input
-----
(2,128,1) tensor containing raw IQ samples

Output
------
Softmax probabilities over modulation classes

Author:
================================================================================
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Dropout,
)


def build_dnn(num_classes: int,
              learning_rate: float = 1e-3,
              dropout: float = 0.5,
              input_shape: tuple[int, int, int] = (2, 128, 1)) -> tf.keras.Model:

    inputs = Input(shape=input_shape, name="Input")

    x = Flatten(
        name="Flatten"
    )(inputs)

    x = Dense(
        units=512,
        activation="relu",
        name="Dense_1"
    )(x)

    x = Dropout(
        rate=dropout,
        name="Dropout_1"
    )(x)

    x = Dense(
        units=256,
        activation="relu",
        name="Dense_2"
    )(x)

    x = Dropout(
        rate=dropout,
        name="Dropout_2"
    )(x)

    x = Dense(
        units=128,
        activation="relu",
        name="Dense_3"
    )(x)

    x = Dropout(
        rate=dropout,
        name="Dropout_3"
    )(x)

    outputs = Dense(
        units=num_classes,
        activation="softmax",
        name="Softmax_Output"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="DNN"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model