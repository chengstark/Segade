import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from keras.layers import *
from keras.models import Model
import numpy as np


np.random.seed(1)
tf.random.set_seed(1)


def encoder_block(x, filter_size, kernel_size):
    """
    SegMADe encoder block
    :param x: x
    :type x: tf.Tensor
    :param filter_size: filter size
    :type filter_size: int
    :param kernel_size: kernel size
    :type kernel_size: int
    :return: return
    :rtype: tf.Tensor
    """

    prev_res = x

    x = Conv1D(filter_size, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(x)
    x = BatchNormalization()(x)

    res = Conv1D(filter_size, kernel_size=1, padding="same")(prev_res)
    x = add([x, res])

    return x


def decoder_block(x, filter_size, kernel_size):
    """
    SegMADe decoder block
    :param x: x
    :type x: tf.Tensor
    :param filter_size: filter size
    :type filter_size: int
    :param kernel_size: kernel size
    :type kernel_size: int
    :return: return
    :rtype: tf.Tensor
    """
    res = x

    x = Conv1D(filter_size, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(x)
    x = BatchNormalization()(x)

    res = Conv1D(filter_size, kernel_size=1, padding="same")(res)
    x = add([x, res])

    return x


def construct_SegMADe(filter_size):
    """
    Construct SegMADe
    :param filter_size: filter size
    :type filter_size: int
    :return: SegMADe model
    :rtype: tf.keras.Model
    """

    ipt = Input((None, 1))
    x = ipt
    encoding_outs = []
    kernel_sizes = [80, 40, 20, 10, 5]

    # encoding path
    for i in range(5):
        x = encoder_block(x, filter_size * (2 ** i), kernel_sizes[i])

        if i == 4:
            x = Dropout(0.2)(x)

        encoding_outs.append(x)

        x = MaxPool1D(pool_size=2, strides=2, padding="same")(x)

    # bottom
    x = Conv1D(filter_size * (2 ** 4), kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # decoding path
    for i in reversed(range(5)):
        x = UpSampling1D(size=2)(x)
        x = Concatenate()([x, encoding_outs[i]])
        x = decoder_block(x, filter_size * (2 ** i), kernel_sizes[i])

    seg_out = Conv1D(1, 1, activation="sigmoid", padding="same", name="seg_out")(x)

    return Model(ipt, seg_out)






