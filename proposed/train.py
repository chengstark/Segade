import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['PYTHONHASHSEED'] = "1"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import Adam
from keras.utils import plot_model
import time
from utils import *
import pickle as pkl
from model import *
from keras import backend as K
from pathlib import Path
import random

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


def AC(y_true, y_pred):
    """
    Active Contour loss, keras custom loss function
    :param y_true: tensor, y_true
    :param y_pred: tensor, y_pred
    :return: tensor, active contour loss
    """
    x = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    delta_x = x[:, :-2, :] ** 2

    transition = K.mean(K.sqrt(delta_x + 1e-6), axis=1)

    c1 = K.ones_like(y_true)
    c2 = K.zeros_like(y_true)
    region_in = K.abs(K.mean(y_pred * ((y_true - c1) ** 2), axis=1))
    region_out = K.abs(K.mean((1 - y_pred) * ((y_true - c2) ** 2), axis=1))

    return 4 * transition + (region_in + region_out)


def generate_sample_weight(y):
    """
    Generate sample weight for training
    :param y: array, ground truth
    :return: array, sample weight
    """
    n0 = np.where(y == 0)[0].shape[0]
    n1 = np.where(y == 1)[0].shape[0]
    print('Good {} | Bad {}'.format(n0, n1))
    sample_weight = np.ones_like(y)

    if n0 > n1:
        sample_weight[y == 1] *= n0 / n1
    elif n0 < n1:
        sample_weight[y == 0] *= n1 / n0

    return sample_weight


def model_train(
    fidx,
    plot_model_to_img=False,
    learning_rate=0.005,
    n_epochs=200,
    batch_size=64,
    es_patience=15,
    shuffle=False
):
    """

    :param fidx: integer, fold index range(0, 10)
    :param plot_model_to_img: bool, plot the model structure to image or not
    :param learning_rate: float, initial learning rate
    :param n_epochs: integer, number of training epochs
    :param batch_size: integer, batch size
    :param es_patience: integer, early stop patience
    :param shuffle: bool, shuffle or not for keras training
    :return: None 
    """
    print('-------------------------------Training Fold {}-------------------------------'.format(fidx))

    def lrs(epoch):
        """
        Keras custom learning rate scheduler
        :param epoch: integer, epoch
        :return: float, learning rate
        """
        if epoch < 15:
            lr = learning_rate
        elif epoch < 35:
            lr = learning_rate / 10
        else:
            lr = learning_rate / 50
        return lr

    start_time = time.time()
    data_dir = str(Path(os.getcwd()).parent) + '/data_folds/new_PPG_DaLiA_train/'
    X_train = np.load(data_dir+'/X_train_{}.npy'.format(fidx))
    X_val = np.load(data_dir+'/X_val_{}.npy'.format(fidx))
    y_seg_train = np.load(data_dir+'/y_seg_train_{}.npy'.format(fidx))
    y_seg_val = np.load(data_dir+'/y_seg_val_{}.npy'.format(fidx))

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    y_seg_train = y_seg_train.reshape((y_seg_train.shape[0], y_seg_train.shape[1], 1))
    y_seg_val = y_seg_val.reshape((y_seg_val.shape[0], y_seg_val.shape[1], 1))

    model_dir = 'model/{}/'.format(fidx)

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    early_stopping = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=1, mode='min')
    mcp_save = ModelCheckpoint(
        model_dir + '/unet_best.h5', save_best_only=True,
        monitor='val_loss', mode='min')

    print(y_seg_train.squeeze().shape)

    sample_weight_train = generate_sample_weight(y_seg_train.squeeze())
    sample_weight_val = generate_sample_weight(y_seg_val.squeeze())


    unet = construct_unet(filter_size=16)

    def dice_metric(y_true, y_pred):
        intersection = K.sum(y_pred * y_true)
        smooth = 0.0001
        dice = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
        return dice

    unet.compile(
        optimizer=Adam(learning_rate=0.0005),
        metrics=[dice_metric],
        loss=AC
    )

    history = unet.fit(
        x=X_train,
        y=y_seg_train,
        epochs=n_epochs,
        shuffle=shuffle,
        verbose=1,
        batch_size=batch_size,
        validation_data=(X_val, y_seg_val, sample_weight_val),
        callbacks=[LearningRateScheduler(lrs), early_stopping, mcp_save],
        sample_weight=sample_weight_train
    )


    if plot_model_to_img:
        plot_model(unet, show_shapes=True, show_layer_names=True,
                   to_file=model_dir + '/model_plot.jpg')

    plot_history(history)
    plt.tight_layout()
    plt.savefig(model_dir + '/plot.jpg')
    plt.clf()
    plt.close('all')

    with open(model_dir + '/hist', 'wb') as file_pi:
        pkl.dump(history.history, file_pi)

    return time.time() - start_time


if __name__ == '__main__':
    for fidx in range(10):
        model_train(
            fidx=fidx,
            plot_model_to_img=True,
            learning_rate=0.0005,
            n_epochs=200,
            batch_size=64
        )



