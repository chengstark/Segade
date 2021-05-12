import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import scipy.stats as stats
from pathlib import Path

np.random.seed(1)


def check_mkdir(dir_):
    """
    Check if the directory exists, if not create this directory
    :param dir_: string, target directory
    :return: None
    """
    if not os.path.isdir(dir_):
        os.mkdir(dir_)


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


def pulse_segmentation(ppg):
    """
    Pulse segmentation
    :param ppg: array, ppg signal
    :return: list, segmented pulses [[start, end], [start, end], [start, end]...]
    """
    pos_peaks, _ = find_peaks(ppg, width=10)
    neg_peaks, _ = find_peaks(1 - ppg, width=10)

    pulses = []
    for pos_peak in pos_peaks:
        prev_neg_peak = 0
        next_neg_peak = 1919

        for idx in reversed(range(0, pos_peak)):
            if idx in neg_peaks:
                prev_neg_peak = idx
                break

        for idx in range(pos_peak, 1920):
            if idx in neg_peaks:
                next_neg_peak = idx
                break

        pulses.append([prev_neg_peak, next_neg_peak])

    return pulses


def create_template(n, fidx):
    """
    Create templates for template matching and save templates
    :param n: integer, number of templates
    :param fidx: integer, fold index range(0, 10)
    :return: None
    """
    if not os.path.isdir('templates/{}/'.format(fidx)):
        os.mkdir('templates/{}/'.format(fidx))

    data_dir_train = str(Path(os.getcwd()).parent) + '/data_folds/'
    X = np.load(data_dir_train + '/new_PPG_DaLiA_train/X_train_{}.npy'.format(fidx)).squeeze()
    y = np.load(data_dir_train + '/new_PPG_DaLiA_train/y_seg_train_{}.npy'.format(fidx)).squeeze()

    templates_pool = []

    for idx, row in enumerate(X):
        pulses = pulse_segmentation(row)

        for pulse_idx, pulse in enumerate(pulses):
            s, e = pulse
            y_seg = y[idx][s:e]
            x_seg = row[s:e]
            x_seg = (x_seg - np.min(x_seg)) / (np.max(x_seg) - np.min(x_seg))

            if np.count_nonzero(y_seg) == 0:
                templates_pool.append(x_seg)

    ref_idices = np.random.choice(np.asarray(list(range(len(templates_pool)))), size=n, replace=False)

    for idx, ref_idx in enumerate(ref_idices):
        np.save('templates/{}/{}.npy'.format(fidx, idx), templates_pool[ref_idx])


def match_template(pulse_seg, fidx):
    """
    Match the templates, calculate the smallest DTW distance
    :param pulse_seg: list, segmented pulses
    :param fidx: integer, fold index range(0, 10)
    :return: integer, minimum distances; integer, index of the templates; lists, templates
    """
    templates = []
    for template_filename in os.listdir('templates'):
        templates.append(np.load('templates/{}/{}.npy'.format(fidx, template_filename)))

    dists = []
    for template in templates:
        distance, path = fastdtw(pulse_seg, template, dist=euclidean)
        dists.append(distance)

    return min(dists), dists.index(min(dists)), templates


def make_predictions(ppg, dtw_thresh, fidx):
    """
    Match the templates, applying threshold on the calculated DTW distances
    :param ppg: array, ppg signals
    :param dtw_thresh: integer, DTW threshold
    :param fidx: integer, fold index range(0, 10)
    :return: array, segmentation prediction labels
    """
    pulses = pulse_segmentation(ppg)
    y_base = np.zeros_like(ppg)

    for pulse in pulses:
        s, e = pulse
        ppg_seg = ppg[s:e]
        ppg_seg = (ppg_seg - np.min(ppg_seg)) / (np.max(ppg_seg) - np.min(ppg_seg))

        distance, matched_idx, templates = match_template(ppg_seg, fidx)

        if distance > dtw_thresh:
            y_base[s:e] = 1

    return y_base
