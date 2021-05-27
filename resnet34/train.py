import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['PYTHONHASHSEED'] = "1"
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

tf.get_logger().setLevel("ERROR")

from model import resnet_34
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from utils import *
from keras.layers import *
from keras import Model
import pickle as pkl
from keras.optimizers import Adam
from scipy.signal import resample
import random
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


def get_tl_model():
    """
    Construct transfer learning Renset 34 model
    :return: tl resnet 34
    :rtype: tf.keras.Model
    """
    res_34 = resnet_34(num_classes=2)
    res_34.load_weights('resnet_34_weight.hdf5')

    layers = [layer for layer in res_34.layers]
    layer_names = [layer.name for layer in res_34.layers]
    print(layer_names[138])
    x = layers[138].output
    x = GlobalAveragePooling1D()(x)
    x = Dense(2, activation='sigmoid')(x)

    custom_model = Model(res_34.input, x)
    for layer in custom_model.layers[:120]:
        layer.trainable = False

    return custom_model


def lrs(epoch):
    """
    Resnet34 learning rate scheduler
    :param epoch: epochs
    :type epoch: int
    :return: learning rate
    :rtype: float
    """
    learning_rate = 0.00001
    if epoch < 10:
        lr = learning_rate
    elif epoch < 50:
        lr = learning_rate / 2
    else:
        lr = learning_rate / 10
    return lr


def generate_sample_weight(y):
    """
    Generate sample weight for training
    :param y: ground truth
    :type y: np.ndarray
    :return: sample weight
    :rtype: np.ndarray
    """
    n0 = np.where(y[:, 1] == 0)[0].shape[0]
    n1 = np.where(y[:, 1] == 1)[0].shape[0]
    print('Good {} | Bad {}'.format(n0, n1))

    sample_weight = np.ones(shape=(y.shape[0],))
    if n0 > n1:
        sample_weight[y[:, 1] == 1] = n0 / n1
    elif n0 < n1:
        sample_weight[y[:, 1] == 0] = n1 / n0

    return sample_weight


def train_res34(fidx, threshold):
    """
    Train the Resnet34
    :param fidx: fold index range(0, 10)
    :type fidx: int
    :param threshold: range(0.0, 1.0, 0.1), > threshold -> artifact, <= threshold -> clean
    :type threshold: float
    :return: None
    :rtype: None
    """
    print('-------------------------------Training Fold {} Thresh {}-------------------------------'.format(fidx,
                                                                                                            threshold))
    model_dir = 'model/thresh_{}/{}'.format(threshold, fidx)

    if not os.path.isdir('model/thresh_{}/'.format(threshold)):
        os.mkdir('model/thresh_{}/'.format(threshold))

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    np.save(model_dir + '/{}'.format(int(threshold * 100)), np.asarray([]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')
    mcp_save = ModelCheckpoint(
        model_dir + '/{}_res34.h5'.format(threshold), save_best_only=True,
        monitor='val_loss', mode='min'
    )

    X_train = np.load('data/X_train_{}_{}.npy'.format(fidx, threshold))
    X_val = np.load('data/X_val_{}_{}.npy'.format(fidx, threshold))
    y_class_train = np.load('data/y_class_train_{}_{}.npy'.format(fidx, threshold))
    y_class_val = np.load('data/y_class_val_{}_{}.npy'.format(fidx, threshold))

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    sample_weight_train = generate_sample_weight(y_class_train)
    sample_weight_val = generate_sample_weight(y_class_val)

    tl_res34 = get_tl_model()
    tl_res34.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )

    history = tl_res34.fit(
        x=X_train,
        y=y_class_train,
        epochs=100,
        shuffle=False,
        verbose=1,
        batch_size=64,
        validation_data=(X_val, y_class_val, sample_weight_val),
        callbacks=[LearningRateScheduler(lrs), early_stopping, mcp_save],
        sample_weight=sample_weight_train
    )

    plot_history(history)
    plt.tight_layout()
    plt.savefig(model_dir + '/train_hist_plot_{}.jpg'.format(threshold))
    plt.clf()
    plt.close('all')

    with open(model_dir + '/train_hist'.format(threshold), 'wb') as file_pi:
        pkl.dump(history.history, file_pi)

    tl_res34.save(model_dir + '/model')


if __name__ == '__main__':
    for thresh in range(0, 10):
            thresh = thresh / 10
            for fidx in range(10):
                train_res34(fidx, thresh)