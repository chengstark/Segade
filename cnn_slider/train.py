import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['PYTHONHASHSEED'] = "1"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
from sklearn.model_selection import train_test_split
from model import *
import pickle as pkl
import numpy as np
import random
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
import time
from pathlib import Path

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


def model_train(
        threshold,
        seg_length,
        epoch,
        fidx
):
    """
    CNN_sliding window classifier training
    :param train_amount: integer, training set size
    :param val_amount: integer validation set size
    :param threshold: float, range(0.0, 1.0, 0.1), > threshold -> artifact, <= threshold -> clean
    :param seg_length: integer, seconds*64Hz sampling rate, segment length/ sliding window length
    :param epoch: integer, number of epoch
    :param fidx: integer, fold index range(0, 10)
    :return: None
    """
    print('---------------------------Training Threshold {} Fold {}---------------------------'.format(threshold,
                                                                                                       fidx))
    model_dir = 'model/thresh_{}_seg_length_{}/{}/'.format(threshold, seg_length, fidx)

    check_mkdir('model/thresh_{}_seg_length_{}/'.format(threshold, seg_length))
    check_mkdir(model_dir)
    check_mkdir(model_dir + '/blind_segs/')

    early_stopping = EarlyStopping(monitor='val_loss', patience=25, verbose=1, mode='min')
    mcp_save = ModelCheckpoint(model_dir + '/slider_best_{}.h5'.format(threshold), save_best_only=True,
                               monitor='val_loss', mode='min'
                               )

    X_train = np.load('data/X_train_{}_{}.npy'.format(threshold, fidx))
    X_val = np.load('data/X_val_{}_{}.npy'.format(threshold, fidx))
    y_train = np.load('data/y_train_{}_{}.npy'.format(threshold, fidx))
    y_val = np.load('data/y_val_{}_{}.npy'.format(threshold, fidx))

    print(X_train.shape)
    # np.save(model_dir + '/blind_segs/blind_seg_X_train.npy', X_train)
    # np.save(model_dir + '/blind_segs/blind_seg_y_train.npy', y_train)
    # np.save(model_dir + '/blind_segs/blind_seg_X_val.npy', X_val)
    # np.save(model_dir + '/blind_segs/blind_seg_y_val.npy', y_val)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    print(X_train.shape, y_train.shape)

    learning_rate = 0.00001

    def lrs(epoch):
        """
        Keras custom learning rate scheduler
        :param epoch: integer, epoch
        :return: float, learning rate
        """
        if epoch < 20:
            lr = learning_rate
        elif epoch < 80:
            lr = learning_rate / 5
        else:
            lr = learning_rate / 10
        return lr

    model = get_model(seg_length)
    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        metrics=["accuracy"],
        loss='binary_crossentropy'
    )

    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=epoch,
        shuffle=True,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[LearningRateScheduler(lrs), early_stopping, mcp_save]
    )

    plot_history(history)
    plt.tight_layout()
    plt.savefig(model_dir + '/plot_{}.jpg'.format(threshold))
    plt.clf()
    plt.close('all')

    with open(model_dir + '/hist_{}'.format(threshold), 'wb') as file_pi:
        pkl.dump(history.history, file_pi)


if __name__ == '__main__':
    epoch = 200
    seg_length = 64 * 3
    interval = 64
    assert 1920 % seg_length == 0
    for threshold in range(0, 10):
        threshold = threshold / 10
        for fidx in range(10):
            model_train(threshold, seg_length, epoch, fidx)
