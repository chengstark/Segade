from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import os


np.random.seed(1)


def perf_measure(y_actual, y_hat):
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

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
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


def check_mkdir(dir_):
    """
    Check if the directory exists, if not create this directory
    :param dir_: target directory
    :type dir_: str
    :return: None
    :rtype: None
    """
    if not os.path.isdir(dir_):
        os.mkdir(dir_)


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
