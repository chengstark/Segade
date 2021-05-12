import numpy as np
from utils import *
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import time
from visualizer import *


def template_eval(dtw_thresh, plot_limiter, fidx, TESTSET_NAME):
    """
    Pulse template matching evaluation
    :param dtw_thresh: integer, dynamic time warping threshold
    :param plot_limiter: integer, number of plots to generate (set to 0 if no plots are needed or to reduce processing time)
    :param fidx: integer, fold index range(0, 10)
    :param TESTSET_NAME: string, test set name, make sure to have this named folder in parent 'data/' directory
    :return: TPR, FPR, DICE score
    """

    working_dir = 'results/{}/{}/dtw_thresh_{}/'.format(TESTSET_NAME, fidx, dtw_thresh)
    plot_dir = working_dir+'/plots/'

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    y_seg_trues = np.load(working_dir+'/y_true_{}.npy'.format(dtw_thresh))
    y_seg_preds = np.load(working_dir+'/y_pred_{}.npy'.format(dtw_thresh))

    data_dir = str(Path(os.getcwd()).parent) + '/data/{}/'.format(TESTSET_NAME)
    X_test = np.load(data_dir+'/processed_dataset/scaled_ppgs.npy')

    report_file = open(working_dir+'/eval_report.txt', 'w+')

    y_pred_flat = y_seg_preds.flatten().astype(np.int8)
    y_true_flat = y_seg_trues.flatten().astype(np.int8)

    print(classification_report(y_true_flat, y_pred_flat, target_names=["0", "1"]), file=report_file)
    print('\n', file=report_file)

    intersection = np.sum(y_pred_flat * y_true_flat)
    smooth = 0.0000001
    dice = (2. * intersection + smooth) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + smooth)
    print('DICE: {}\n'.format(dice), file=report_file)

    report_file.close()

    TPR, FPR = calc_TPR_FPR(y_true_flat, y_pred_flat)

    n_transitions = 0
    for pred in y_seg_preds:
        n_transitions += len(get_edges(pred.flatten()))
    n_transitions /= y_seg_preds.shape[0]

    if plot_limiter > 0:
        pbar = tqdm(X_test[:plot_limiter, :], position=0, leave=False)
        idx = 0
        for ppg in pbar:
            plot_result(
                ppg, y_seg_trues[idx], y_seg_preds[idx], show=False, save=True,
                save_path=plot_dir + '{}.jpg'.format(idx)
            )
            idx += 1

    return TPR, FPR, dice, n_transitions






