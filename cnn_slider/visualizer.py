import matplotlib.pyplot as plt
import matplotlib
import sys
import numpy as np
import shutil
from tqdm import tqdm


def get_edges(label):
    """
    Get edges of segmentation labels
    :param label: array. segmentation label
    :return: list, edges of segmentation labels
    """
    label = label.flatten()
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


def plot_result(ppg, true_label, pred_label=None, show=True, prob=None, save=True,
                save_path=None, plot_prob=False, plot_true_only=False):
    """
    Segmentation result visualizer
    :param ppg: array, ppg signals
    :param true_label: array, segmentation true label
    :param pred_label: array, segmentation prediction
    :param show: bool, display the plot or not
    :param prob: array, prediction probabilities or cams
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
                plot_on_ax(ppg, pred_label, ax2, title='ResNet 34 CAM Label', color='darkorange')

                cmap = matplotlib.cm.get_cmap('rainbow')
                ax4.plot(ppg)
                ax4.margins(x=0, y=0)

                for s, prob in enumerate(prob):
                    ax4.axvline(x=s, color=cmap(prob), alpha=0.2)

                norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array(np.asarray([]))
                fig.colorbar(sm, ax=ax4, orientation='horizontal')
            else:
                fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(30, 3))
                plot_on_ax(ppg, true_label, ax1, title='Human Label')
                plot_on_ax(ppg, true_label, ax3, title='Overlay', overlay=True, label2=pred_label, color2='darkorange')
                plot_on_ax(ppg, pred_label, ax2, title='UNet Label', color='darkorange')

    if show:
        plt.show()

    if save:
        plt.savefig(save_path, facecolor='white')
        fig.clear()
        plt.close('all')
