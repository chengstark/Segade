import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os


n_fold = 10

kf = KFold(n_splits=n_fold, shuffle=True, random_state=1)

TRAINING_SET_NAME = 'new_PPG_DaLiA_train'
# TRAINING_SET_NAME = 'capnobase'

if not os.path.exists('data_folds/{}/'.format(TRAINING_SET_NAME)):
    os.mkdir('data_folds/{}/'.format(TRAINING_SET_NAME))

X = np.load('data/{}/processed_dataset/scaled_ppgs.npy'.format(TRAINING_SET_NAME))
y_seg = np.load('data/{}/processed_dataset/seg_labels.npy'.format(TRAINING_SET_NAME))

for fidx, splits in enumerate(kf.split(X)):
    train_idx, val_idx = splits
    X_train, X_val = X[train_idx], X[val_idx]
    y_seg_train, y_seg_val = y_seg[train_idx], y_seg[val_idx]

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    y_seg_train = y_seg_train.reshape(y_seg_train.shape[0], y_seg_train.shape[1], 1)
    y_seg_val = y_seg_val.reshape(y_seg_val.shape[0], y_seg_val.shape[1], 1)

    np.save('data_folds/{}/X_train_{}.npy'.format(TRAINING_SET_NAME, fidx), X_train)
    np.save('data_folds/{}/X_val_{}.npy'.format(TRAINING_SET_NAME, fidx), X_val)
    np.save('data_folds/{}/y_seg_train_{}.npy'.format(TRAINING_SET_NAME, fidx), y_seg_train)
    np.save('data_folds/{}/y_seg_val_{}.npy'.format(TRAINING_SET_NAME, fidx), y_seg_val)

    print('Created fold {} save file'.format(fidx))

