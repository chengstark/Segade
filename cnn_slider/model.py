import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
from keras.layers import *
from utils import *
import pickle as pkl
import keras
from keras.models import Model
import numpy as np

np.random.seed(1)
tf.random.set_seed(1)


def get_model(seg_length):
    """
    Construct model for CNN sliding window method
    :param seg_length: integer, sliding window length
    :return: keras model
    """
    ipt = Input((seg_length, 1))
    x = Conv1D(filters=64, kernel_size=10, activation='relu')(ipt)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3)(x)

    x = Conv1D(filters=64, kernel_size=5, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3)(x)

    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3)(x)

    x = Dropout(0.15)(x)
    x = Flatten()(x)

    out = Dense(units=1, activation='sigmoid')(x)

    model = Model(ipt, out)
    return model

