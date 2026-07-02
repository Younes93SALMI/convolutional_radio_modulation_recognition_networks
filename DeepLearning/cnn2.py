"""
================================================================================
VT-CNN2
--------------------------------------------------------------------------------
Implementation of the VT-CNN2 architecture proposed in

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
    ZeroPadding2D,
)


def build_cnn2(num_classes: int,
                 learning_rate: float = 1e-3,
                 dropout: float = 0.6,
                 input_shape: tuple[int, int, int] = (2, 128, 1)) -> tf.keras.Model:

    inputs = Input(shape=input_shape, name="Input")

    x = ZeroPadding2D(
        padding=(0, 2),
        name="ZeroPadding_1"
    )(inputs)

    x = Conv2D(
        filters=256,
        kernel_size=(1, 3),
        activation="relu",
        padding="same",
        name="Conv2D_1"
    )(x)

    x = Dropout(
        rate=dropout,
        name="Dropout_1"
    )(x)

    x = ZeroPadding2D(
        padding=(0, 2),
        name="ZeroPadding_2"
    )(x)

    x = Conv2D(
        filters=80,
        kernel_size=(2, 3),
        activation="relu",
        padding="valid",
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
        units=256,
        activation="relu",
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
        name="VT-CNN2"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
