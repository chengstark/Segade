import numpy as np
from pathlib import Path
import os
from scipy.signal import resample
from tqdm import tqdm
from utils import *


def process_data(fidx, threshold, TESTSET_NAME, process_test_only=False):
    """
    Resnet34 pre processing data
    :param fidx: integer, fold index range(0, 10)
    :param threshold: float, range(0.0, 1.0, 0.1), > threshold -> artifact, <= threshold -> clean
    :param TESTSET_NAME: string, test set name, make sure to have this named folder in parent 'data/' directory
    :param process_test_only: bool, process only testset. Set to true when processing independent test sets
    :return: None
    """

    data_dir_train = str(Path(os.getcwd()).parent) + '/data_folds/'
    X_train = np.load(data_dir_train + '/new_PPG_DaLiA_train/X_train_{}.npy'.format(fidx))
    X_val = np.load(data_dir_train + '/new_PPG_DaLiA_train/X_val_{}.npy'.format(fidx))
    y_seg_train = np.load(data_dir_train + '/new_PPG_DaLiA_train/y_seg_train_{}.npy'.format(fidx))
    y_seg_val = np.load(data_dir_train + '/new_PPG_DaLiA_train/y_seg_val_{}.npy'.format(fidx))

    if not process_test_only:

        # print(X_train.shape, X_val.shape, y_seg_train.shape, y_seg_val.shape)
        y_class_train = []
        y_class_val = []
        for row in y_seg_train:
            n1 = np.count_nonzero(row)
            perc = n1 / row.shape[0]
            if perc > threshold:
                y_class_train.append([0, 1])
            else:
                y_class_train.append([1, 0])

        for row in y_seg_val:
            n1 = np.count_nonzero(row)
            perc = n1 / row.shape[0]
            if perc > threshold:
                y_class_val.append([0, 1])
            else:
                y_class_val.append([1, 0])

        X_train_upsampled = []
        X_val_upsampled = []

        for row in X_train:
            X_train_upsampled.append(resample(row, 7201))

        for row in X_val:
            X_val_upsampled.append(resample(row, 7201))

        X_train_upsampled = np.asarray(X_train_upsampled)
        X_val_upsampled = np.asarray(X_val_upsampled)
        y_class_train = np.asarray(y_class_train)
        y_class_val = np.asarray(y_class_val)

        np.save('data/X_train_{}_{}.npy'.format(fidx, threshold), X_train_upsampled)
        np.save('data/X_val_{}_{}.npy'.format(fidx, threshold), X_val_upsampled)
        np.save('data/y_class_train_{}_{}.npy'.format(fidx, threshold), y_class_train)
        np.save('data/y_class_val_{}_{}.npy'.format(fidx, threshold), y_class_val)
        np.save('data/y_seg_train_{}_{}.npy'.format(fidx, threshold), y_seg_train)
        np.save('data/y_seg_val_{}_{}.npy'.format(fidx, threshold), y_seg_val)

    shap_backgrounds = []
    for idx, row in enumerate(X_train):
        y_seg = y_seg_train[idx]
        n1 = np.count_nonzero(y_seg)
        if n1 == 0:
            shap_backgrounds.append(resample(row, 7201))
    shap_backgrounds = np.asarray(shap_backgrounds)
    np.save('data/shap_background_{}.npy'.format(fidx), shap_backgrounds)

    check_mkdir('data/{}/'.format(TESTSET_NAME))
    data_dir_test = str(Path(os.getcwd()).parent) + '/data/{}/'.format(TESTSET_NAME)
    X_test = np.load(data_dir_test + '/processed_dataset/scaled_ppgs.npy')
    y_seg_test = np.load(data_dir_test + '/processed_dataset/seg_labels.npy')
    X_test_upsampled = []
    y_class_test = []
    for row in X_test:
        X_test_upsampled.append(resample(row, 7201))

    for row in y_seg_test:
        n1 = np.count_nonzero(row)
        perc = n1 / row.shape[0]
        if perc > threshold:
            y_class_test.append([0, 1])
        else:
            y_class_test.append([1, 0])
    X_test_upsampled = np.asarray(X_test_upsampled)
    y_class_test = np.asarray(y_class_test)
    np.save('data/{}/X_test_{}.npy'.format(TESTSET_NAME, threshold), X_test_upsampled)
    np.save('data/{}/y_class_test_{}.npy'.format(TESTSET_NAME, threshold), y_class_test)
    np.save('data/{}/y_seg_test_{}.npy'.format(TESTSET_NAME, threshold), y_seg_test)


if __name__ == '__main__':
    pbar1 = tqdm(range(10))
    for fidx in pbar1:
        pbar1.set_description('Running fold {}'.format(fidx))
        pbar2 = tqdm(range(0, 11), leave=False)
        for thresh in pbar2:
            thresh = thresh / 10
            pbar2.set_description('Running threshold {}'.format(thresh))
            process_data(fidx, thresh, '', process_test_only=False)

