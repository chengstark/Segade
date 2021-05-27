import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import math
import random
import os

np.random.seed(1)
random.seed(1)


def perf_measure(y_true, y_pred):
    """
    Calculate TP, FP, TN, FN
    :param y_true: true labels
    :type y_true: np.ndarray
    :param y_pred: predicted labels
    :type y_pred: np.ndarray
    :return: TP, FP, TN, FN
    :rtype: int, int, int, int
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_true[i] != y_pred[i]:
            FP += 1
        if y_true[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_true[i] != y_pred[i]:
            FN += 1

    return TP, FP, TN, FN


def calc_TPR_FPR(y_trues_flat, y_preds_flat):
    """
    Calculate TPR and FPR
    :param y_trues_flat: true labels
    :type y_trues_flat: np.ndarray
    :param y_preds_flat: predicted labels
    :type y_preds_flat: np.ndarray
    :return: TPR, FPR
    :rtype: float
    """
    TP, FP, TN, FN = perf_measure(y_trues_flat, y_preds_flat)
    if TP == 0:
        TPR = 0.0
    else:
        TPR = TP / (TP + FN)
    if FP == 0:
        FPR = 0.0
    else:
        FPR = FP / (FP + TN)
    return TPR, FPR


def check_mkdir(dir_):
    """
    Check if the directory exists, if not create this directory
    :param dir_: target directory
    :type dir_: str
    :return: None
    :rtype: None
    """
    if not os.path.exists(dir_):
        os.mkdir(dir_)


def sort_TPRs_FPRs(TPR, FPR):
    """
    Sort TPR by FPR
    :param TPR: TPRs
    :type TPR: list(float)
    :param FPR: FPRs
    :type FPR: list(float)
    :return: sorted_TPR, sorted_FPR
    :rtype: list(float), list(float)
    """
    sorted_TPR = [x for _, x in sorted(zip(FPR.tolist(), TPR.tolist()))]
    sorted_FPR = [x for x, _ in sorted(zip(FPR.tolist(), TPR.tolist()))]
    return sorted_TPR, sorted_FPR


def plot_confusion_matrix(cm, index=[0, 1, 2], columns=[0, 1, 2]):
    """
    Plot confusion matrix
    :param cm: confusion matrix
    :type cm: np.ndarray
    :param index: index
    :type index: list
    :param columns: columns
    :type columns: list
    :return: None
    :rtype: None
    """
    df_cm = pd.DataFrame(cm, index=index,
                         columns=columns)
    plt.figure(figsize=(6, 5))
    sn.heatmap(df_cm, annot=True)


def plot_history(history):
    """
    Plot keras training history
    :param history: keras history
    :type history: tf.keras.callbacks.History
    :return: None
    :rtype: None
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')


def blind_seg(X, y, amount, seg_length, threshold):
    """
    Blind segmentation, randomly generate segments/ windows from signal and signal segmentation label
    :param X: signal
    :type X: np.ndarray
    :param y: segmentation label
    :type y: np.ndarray
    :param amount: number of windows to generate
    :type amount: integer
    :param seg_length: sliding window length
    :type seg_length: int
    :param threshold: range(0.0, 1.0, 0.1), > threshold -> artifact, <= threshold -> clean
    :type threshold: float
    :return: windows and their corresponding classification labels (clean/ artifacts)
    :rtype: list(np.ndarray), list(np.ndarray)
    """

    cleans = []
    artifacts = []
    for i in range(X.shape[0]):
        X_row = X[i]
        y_row = y[i]
        starts = np.random.choice(range(X_row.shape[0] - seg_length), X_row.shape[0] // 5, replace=False)

        for start in starts:
            X_seg = X_row[start: start+seg_length]
            y_seg = y_row[start: start+seg_length]
            n1 = np.count_nonzero(y_seg)
            X_seg = (X_seg - np.min(X_seg)) / (np.max(X_seg) - np.min(X_seg))
            if (n1 / seg_length) > threshold:
                artifacts.append(X_seg)
            else:
                cleans.append(X_seg)

    cleans = np.asarray(cleans)
    artifacts = np.asarray(artifacts)

    if len(cleans) > len(artifacts):
        cleans = cleans[np.random.choice(cleans.shape[0], artifacts.shape[0], replace=False)]
    else:
        artifacts = artifacts[np.random.choice(artifacts.shape[0], cleans.shape[0], replace=False)]

    np.random.shuffle(cleans)
    np.random.shuffle(artifacts)

    cleans = cleans[:amount // 2]
    artifacts = artifacts[:amount // 2]

    X_outs = np.concatenate((cleans, artifacts))
    y_outs = np.concatenate((np.zeros(amount // 2, ), np.ones(amount // 2, )))
    print(cleans.shape, artifacts.shape, X_outs.shape, y_outs.shape)

    return X_outs, y_outs


def process_sliding_results(y_slide, y_seg, slices, prob_thresh):
    """
    Process sliding windows classification results, convert classification labels to segmentation label
    :param y_slide: classification labels
    :type y_slide: np.ndarray
    :param y_seg: segmentation labels (ground truth) for shape reference, passing in the signal would still work
    :type y_seg: np.ndarray
    :param slices: slices of windows, [[start, end], [start, end], [start, end]...]
    :type slices: np.ndarray or list
    :param prob_thresh: probability threshold for calculate ROC
    :type prob_thresh: float
    :return: segmentation mask
    :rtype: np.ndarray
    """
    y_base = np.zeros((y_seg.shape[0], ))
    for idx, y in enumerate(y_slide):
        if y >= prob_thresh:
            y_base[slices[idx][0]:slices[idx][1]] = 1

    return y_base


def slide_window(X, y, seg_length, interval, threshold):
    """
    Create sliding windows based on segment/ window length and window interval
    :param X: signal
    :type X: np.ndarray
    :param y: segmentation label ground truth
    :type y: np.ndarray
    :param seg_length: seconds*64Hz sampling rate, segment length/ sliding window length
    :type seg_length: int
    :param interval: seconds*64Hz sampling rate, interval between sliding windows
    :type interval: int
    :param threshold: range(0.0, 1.0, 0.1), > threshold -> artifact, <= threshold -> clean
    :type threshold: float
    :return: signal windows outputs, window classification labels, slices of windows, [[start, end], [start, end], [start, end]...]
    :rtype: np.ndarray, np.ndarray, list[int]
    """
    X_outs = []
    y_outs = []
    slices = []
    for start in np.arange(0, X.shape[0] - seg_length + interval, interval):
        s = start
        e = s + seg_length
        X_seg = X[s:e]
        X_seg = (X_seg - np.min(X_seg)) / (np.max(X_seg) - np.min(X_seg))

        X_outs.append(X_seg)
        y_seg = y[s:e]

        n1 = np.count_nonzero(y_seg)

        if (n1 / seg_length) > threshold:
            y_outs.append(1)
        else:
            y_outs.append(0)

        slices.append([s, e])

    return np.asarray(X_outs), np.asarray(y_outs), slices






