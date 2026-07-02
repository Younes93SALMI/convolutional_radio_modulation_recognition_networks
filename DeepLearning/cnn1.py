"""
================================================================================
CNN
--------------------------------------------------------------------------------
Implementation of the CNN architecture proposed in

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
    Conv2D,
    Dense,
    Flatten,
    Dropout,
)
from tensorflow.keras import regularizers


def build_cnn(num_classes: int,
              learning_rate: float = 1e-3,
              dropout: float = 0.5,
              input_shape: tuple[int, int, int] = (2, 128, 1),
              l2_weight: float = 1e-3,
              l1_dense_activity: float = 1e-5) -> tf.keras.Model:

    inputs = Input(shape=input_shape, name="Input")

    x = Conv2D(
        filters=64,
        kernel_size=(1, 3),
        activation="relu",
        padding="valid",
        kernel_regularizer=regularizers.l2(l2_weight),
        name="Conv2D_1"
    )(inputs)

    x = Dropout(
        rate=dropout,
        name="Dropout_1"
    )(x)

    x = Conv2D(
        filters=16,
        kernel_size=(2, 3),
        activation="relu",
        padding="valid",
        kernel_regularizer=regularizers.l2(l2_weight),
        name="Conv2D_2"
    )(x)

    x = Dropout(
        rate=dropout,
        name="Dropout_2"
    )(x)

    x = Flatten(
        name="Flatten"
    )(x)

    x = Dense(
        units=128,
        activation="relu",
        activity_regularizer=regularizers.l1(l1_dense_activity),
        name="Dense_1"
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
        name="CNN"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model