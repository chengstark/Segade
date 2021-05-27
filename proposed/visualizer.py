import matplotlib.pyplot as plt
import matplotlib
import sys
from model import *
import shutil
from tqdm import tqdm

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def get_edges(label):
    """
    Get edges of segmentation labels
    :param label: segmentation label
    :type label: np.ndarray
    :return: edges of segmentation labels
    :rtype: list(int, int)
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


def plot_result(ppg, true_label, pred_label=None, show=True, raw_prediction=None, save=True,
                save_path=None, plot_prob=False, plot_true_only=False, additional_title=''):
    """
    Segmentation result visualizer
    :param ppg: ppg signals
    :type ppg: np.ndarray
    :param true_label: segmentation true label
    :type true_label: np.ndarray
    :param pred_label:  segmentation prediction label
    :type pred_label: np.ndarray
    :param show: display the plot or not
    :type show: bool
    :param prob: raw model output
    :type prob: np.ndarray
    :param save: save the plot or not
    :type save: bool
    :param save_path: save path
    :type save_path: str
    :param plot_prob: plot raw output or not
    :type plot_prob: bool
    :param plot_true_only: plot true labels only or not
    :type plot_true_only: bool
    :return: None
    :rtype: None
    """
    def plot_on_ax(ppg, label, ax, title='', color='g', overlay=False, label2=None, color2='y'):
        """
        Plot signal and label on axis
        :param ppg: ppg signals
        :type ppg: np.ndarray
        :param label: segmentation label
        :type label: np.ndarray
        :param ax: axis to be plotted
        :type ax: matplotlib.axes
        :param title: title
        :type title: str
        :param color: matplotlib color for true label
        :type color: str0
        :param overlay:
        :type overlay: bool
        :param label2: segmentation label
        :type label2: np.ndarray
        :param color2: matplotlib color for predicted label
        :type color2: str
        :return: None
        :rtype: None
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
            plot_on_ax(ppg, true_label, ax, title='Human Label'+' '+additional_title)
        else:
            if plot_prob:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(30, 10))
                plot_on_ax(ppg, true_label, ax1, title='Human Label'+' '+additional_title)
                plot_on_ax(ppg, true_label, ax3, title='Overlay', overlay=True, label2=pred_label, color2='darkorange')
                # ax3.plot(raw_prediction)
                # ax3.set_title('Prediction Time Series')
                plot_on_ax(ppg, pred_label, ax2, title='SegMADe Label'+' '+additional_title, color='darkorange')

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
                plot_on_ax(ppg, pred_label, ax2, title='SegMADe Label'+' '+additional_title, color='darkorange')

    if show:
        plt.show()

    if save:
        plt.savefig(save_path, facecolor='white')
        fig.clear()
        plt.close('all')


def visualize(plot_prob):
    """
    Visualize some samples and results
    :param plot_prob: plot raw model prediction or not
    :type plot_prob: bool
    :return: None
    :rtype: None
    """
    working_dir = 'results'
    plot_dir = working_dir + '/plots/'

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    else:
        shutil.rmtree(plot_dir)
        os.mkdir(plot_dir)

    X_val = np.load(working_dir + '/processed_dataset/scaled_ppgs.npy')
    y_true = np.load(working_dir + '/processed_dataset/seg_labels.npy')

    SegMADe = construct_SegMADe(filter_size=16)
    SegMADe.load_weights(working_dir + '/model_weights/SegMADe_best.h5')
    y_pred = SegMADe.predict(X_val).squeeze()
    y_pred_raw = y_pred
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1

    X_val_plot_bar = tqdm(X_val, position=0, leave=False)

    idx = 0
    for ppg in X_val_plot_bar:
        plot_result(
            ppg, y_true[idx], y_pred[idx], show=False, raw_prediction=y_pred_raw[idx], save=True,
            save_path=plot_dir + '{}.jpg'.format(idx), plot_prob=bool(plot_prob)
        )
        idx += 1


if __name__ == '__main__':
    visualize(plot_prob=False)
