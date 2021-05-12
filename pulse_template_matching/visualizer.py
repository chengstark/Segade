import matplotlib.pyplot as plt
import matplotlib
import sys
import shutil
from tqdm import tqdm
import numpy as np
import os
from scipy.signal import find_peaks


def get_edges(label):
    """
    Get edges of segmentation labels
    :param label: array. segmentation label
    :return: list, edges of segmentation labels
    """
    label = label.flatten( )
    ref = label[1:] - label[:-1]
    base = np.zeros_like(label)
    base[np.where(ref == 1)[0]+1] = 1
    base[np.where(ref == -1)[0]] = 1
    if label[0] == 1:
        base[0] = 1
    if label[-1] == 1:
        base[-1] = 1

    cp_prev = np.concatenate((np.asarray([0]), label[:-1]))
    cp_next = np.concatenate((label[1:], np.asarray([0])))

    cp = cp_prev + cp_next
    cp_base = label - cp

    cp_base[cp_base != 1] = 0
    base[cp_base == 1] = 0

    edges = np.where(base == 1)[0]

    edges = edges.reshape((edges.shape[0] // 2, 2))

    for i in np.where(cp_base == 1)[0]:
        edges = np.concatenate((np.asarray([[i, i]]), edges), axis=0)

    return edges


def plot_result(ppg, true_label, pred_label=None, show=True, raw_prediction=None, save=True,
                save_path=None, plot_prob=False, plot_true_only=False):
    """
    Segmentation result visualizer
    :param ppg: array, ppg signals
    :param true_label: array, segmentation true label
    :param pred_label: array, segmentation prediction
    :param show: bool, display the plot or not
    :param raw_prediction: array, prediction probabilities or cams
    :param save: bool, save the plot or not, if save is true, save path should be specified
    :param save_path: string, save path
    :param plot_prob: bool, plot probability or not
    :param plot_true_only: bool, only plot ground truth or not
    :return: None
    """
    def plot_on_ax(ppg, label, ax, title='', color='g', overlay=False, label2=None, color2='y'):
        """
        Plot signal and label on axis
        :param ppg: array, ppg signals
        :param label: array, segmentation label
        :param ax: matplotlib axis
        :param title: string, title
        :param color: matplotlib color
        :param overlay: bool, overlay ground truth on predictions or not
        :param label2: array, second label array. If overlay is true, this must be specified
        :param color2: matplotlib color for the second label. If overlay is true, this must be specified
        :return: None
        """
        ax.plot(ppg)
        ax.margins(x=0, y=0)

        pos_peaks, _ = find_peaks(ppg, width=10)
        neg_peaks, _ = find_peaks(1 - ppg, width=10)
        ax.plot(pos_peaks, ppg[pos_peaks], "x", c='r')
        ax.plot(neg_peaks, ppg[neg_peaks], "x", c='g')

        edges = get_edges(label)
        if len(edges) > 0:
            for edge in edges:
                s, e = edge
                if s == e:
                    ax.axvline(x=s, color=color, alpha=0.5)
                else:
                    ax.axvspan(xmin=s, xmax=e, facecolor=color, alpha=0.5)
        if overlay:
            edges2 = get_edges(label2)
            if len(edges2) > 0:
                for edge in edges2:
                    s, e = edge
                    if s == e:
                        ax.axvline(x=s, color=color2, alpha=0.5)
                    else:
                        ax.axvspan(xmin=s, xmax=e, facecolor=color2, alpha=0.5)
        ax.set_title(title)

    plt.clf()

    if plot_true_only:
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 3))
        plot_on_ax(ppg, true_label, ax1, title='')
    else:
        if pred_label is None:
            true_label = true_label.flatten()
            fig, ax = plt.subplots()
            plot_on_ax(ppg, true_label, ax, title='Human Label')
        else:
            if plot_prob:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(30, 10))
                plot_on_ax(ppg, true_label, ax1, title='Human Label')
                plot_on_ax(ppg, true_label, ax3, title='Overlay', overlay=True, label2=pred_label, color2='darkorange')
                plot_on_ax(ppg, pred_label, ax2, title='Template Matching', color='darkorange')

                cmap = matplotlib.cm.get_cmap('rainbow')
                ax4.plot(ppg)
                ax4.margins(x=0, y=0)

                for s, prob in enumerate(raw_prediction):
                    ax4.axvline(x=s, color=cmap(prob), alpha=0.2)

                norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array(np.asarray([]))
                fig.colorbar(sm, ax=ax4, orientation='horizontal')
            else:
                fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(30, 3))
                plot_on_ax(ppg, true_label, ax1, title='Human Label')
                plot_on_ax(ppg, true_label, ax3, title='Overlay', overlay=True, label2=pred_label, color2='darkorange')
                plot_on_ax(ppg, pred_label, ax2, title='Template Matching', color='darkorange')

    if show:
        plt.show()

    if save:
        plt.savefig(save_path, facecolor='white')
        fig.clear()
        plt.close('all')


def visualize(dtw_thresh):
    """
    Visualize examples with different DTW thresholds
    :param dtw_thresh: integer, DTW thresholds
    :return: None
    """
    working_dir = 'dtw_thresh_{}/'.format(dtw_thresh)

    plot_dir = working_dir + '/plots/'

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    else:
        shutil.rmtree(plot_dir)
        os.mkdir(plot_dir)

    X_val = np.load('PPG_DaLiA_test/processed_dataset/scaled_ppgs.npy')
    y_true = np.load('PPG_DaLiA_test/processed_dataset/seg_labels.npy')
    y_pred = np.load(working_dir+'y_seg_preds/y_pred_{}.npy'.format(dtw_thresh))

    X_val_plot_bar = tqdm(X_val, position=0, leave=False)

    idx = 0
    for ppg in X_val_plot_bar:
        plot_result(
            ppg, y_true[idx], y_pred[idx], show=False, save=True,
            save_path=plot_dir + '{}.jpg'.format(idx), plot_prob=False
        )
        idx += 1
