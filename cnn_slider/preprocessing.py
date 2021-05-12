from utils import *
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm


def preprocess(train_amount, threshold, seg_length):
    data_dir = str(Path(os.getcwd()).parent) + '/data_folds/new_PPG_DaLiA_train/'
    for fidx in range(10):
        X_train = np.load(data_dir + '/X_train_{}.npy'.format(fidx))
        y_train = np.load(data_dir + '/y_seg_train_{}.npy'.format(fidx))
        X_val = np.load(data_dir + '/X_val_{}.npy'.format(fidx))
        y_val = np.load(data_dir + '/y_seg_val_{}.npy'.format(fidx))
        X_segs_train, y_segs_train = blind_seg(X_train, y_train, train_amount, seg_length, threshold)
        X_segs_val, y_segs_val = blind_seg(X_val, y_val, int(train_amount*0.2), seg_length, threshold)

        np.save('data/X_train_{}_{}.npy'.format(threshold, fidx), X_segs_train)
        np.save('data/X_val_{}_{}.npy'.format(threshold, fidx), X_segs_val)
        np.save('data/y_train_{}_{}.npy'.format(threshold, fidx), y_segs_train)
        np.save('data/y_val_{}_{}.npy'.format(threshold, fidx), y_segs_val)


if __name__ == '__main__':
    train_amount = 5000
    val_amount = int(train_amount * (0.2 / 0.75))
    epoch = 200
    seg_length = 64 * 3
    interval = 64
    assert 1920 % seg_length == 0
    for threshold in tqdm(range(0, 10)):
        threshold = threshold / 10
        preprocess(train_amount, threshold, seg_length)
