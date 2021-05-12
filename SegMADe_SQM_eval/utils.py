import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import os

np.random.seed(1)


def perf_measure(y_actual, y_hat):
    """
    Calculate TP, FP, TN, FN
    :param y_true: numpy array or list, true labels
    :param y_pred: numpy array or list, predicted labels
    :return: integers, TP, FP, TN, FN
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
    sorted_TPR = [x for _, x in sorted(zip(FPR.tolist(), TPR.tolist()))]
    sorted_FPR = [x for x, _ in sorted(zip(FPR.tolist(), TPR.tolist()))]
    return sorted_TPR, sorted_FPR


def check_mkdir(dir_):
    """
    Check if the directory exists, if not create this directory
    :param dir_: string, target directory
    :return: None
    """
    if not os.path.isdir(dir_):
        os.mkdir(dir_)


def plot_confusion_matrix(cm, index=[0, 1, 2], columns=[0, 1, 2]):
    """
    Plot confusion matrix
    :param cm: array, confusion matrix
    :param index: list, index
    :param columns: list, columns
    :return: None
    """
    df_cm = pd.DataFrame(cm, index=index,
                         columns=columns)
    plt.figure(figsize=(6, 5))
    sn.heatmap(df_cm, annot=True)


def shrink_data(X, n_th=3):
    """
    Brute force shrink a signal, delete every n th element
    :param X: array, signal
    :param n_th: integer, n th element to remove
    :return: array, processed signal
    """
    deleted = np.delete(X, np.arange(0, X.shape[1], n_th), axis=1)
    return deleted


def plot_history(history):
    """
    Plot keras training history
    :param history: keras history
    :return: None
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    ax1.plot(history.history['dice_metric'])
    ax1.plot(history.history['val_dice_metric'])
    ax1.set_title('model DICE')
    ax1.set_ylabel('DICE')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')


def plot_history_multi_out(history):
    """
    Plot keras training history
    :param history: keras history
    :return: None
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    ax1.plot(history.history['seg_out_dice_metric'], label='seg_out_dice_metric')
    ax1.plot(history.history['val_seg_out_dice_metric'],label='val_seg_out_dice_metric')
    ax1.set_title('model DICE')
    ax1.set_ylabel('DICE')
    ax1.set_xlabel('epoch')
    ax1.legend(loc='upper left')

    ax2.plot(history.history['class_out_auc'], label='class_out_auc')
    ax2.plot(history.history['val_class_out_auc'], label='val_class_out_auc')
    ax2.set_title('model auc')
    ax2.set_ylabel('auc')
    ax2.set_xlabel('epoch')
    ax2.legend(loc='upper left')

    ax3.plot(history.history['seg_out_loss'], label='seg_out_loss')
    ax3.plot(history.history['val_seg_out_loss'], label='val_seg_out_loss')
    ax3.plot(history.history['class_out_loss'], label='class_out_loss')
    ax3.plot(history.history['val_class_out_loss'], label='val_class_out_loss')
    ax3.set_title('model losses')
    ax3.set_ylabel('losses')
    ax3.set_xlabel('epoch')
    ax3.legend(loc='upper left')

