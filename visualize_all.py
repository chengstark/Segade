import numpy as np
from proposed.utils import *
from tqdm import tqdm
import sys
from scipy.signal import find_peaks
from multiprocessing import Pool


TESTSET_NAME = sys.argv[1]
check_mkdir('visualize_all/{}/'.format(TESTSET_NAME))

X_test = np.load('data/{}/processed_dataset/scaled_ppgs.npy'.format(TESTSET_NAME))
y_seg_true = np.load('data/{}/processed_dataset/seg_labels.npy'.format(TESTSET_NAME))

cnn_slider_result = 'cnn_slider/results/{}/thresh_0.0_seg_length_192/{}/'.format(TESTSET_NAME, 0) + 'y_seg_preds/y_pred_0.0_0.5.npy'
pulsetm_result = 'pulse_template_matching/results/{}/{}/dtw_thresh_1/'.format(TESTSET_NAME, 0)+'/y_pred_1.npy'
res34_cam_result = 'resnet34/results/{}/thresh_{}/{}/'.format(TESTSET_NAME, 0.0, 0)+'y_seg_preds_{}_{}_{}_{}.npy'.format('cam', 0, 0.0, 0.0)
res34_shap_result = 'resnet34/results/{}/thresh_{}/{}/'.format(TESTSET_NAME, 0.0, 0)+'y_seg_preds_{}_{}_{}_{}.npy'.format('shap', 0, 0.0, 0.0)
SegMADe = 'proposed/results/{}/{}/'.format(TESTSET_NAME, 0) + '/y_pred.npy'
# print(cnn_slider_result, os.path.exists(cnn_slider_result))
if os.path.exists(cnn_slider_result):
    y_seg_cnnslider = np.load(cnn_slider_result)
    cnnslider_title = 'CNN Sliding Window Label'
else:
    y_seg_cnnslider = np.zeros_like(y_seg_true)
    cnnslider_title = 'NO CNN Sliding Window Label'

if os.path.exists(pulsetm_result):
    y_seg_pulse_tm = np.load(pulsetm_result)
    pulsetm_title = 'Pulse Template Matching Label'
else:
    y_seg_pulse_tm = np.zeros_like(y_seg_true)
    pulsetm_title = 'NO Pulse Template Matching Label'

if os.path.exists(res34_cam_result):
    y_seg_res34_cam = np.load(res34_cam_result)
    res34_cam_title = 'Resnet34 Grad CAM Label'
else:
    y_seg_res34_cam = np.zeros_like(y_seg_true)
    res34_cam_title = 'NO Resnet34 Grad CAM Label'


if os.path.exists(res34_shap_result):
    y_seg_res34_shap = np.load(res34_shap_result)
    res34_shap_title = 'Resnet34 SHAP Label'
else:
    y_seg_res34_shap = np.zeros_like(y_seg_true)
    res34_shap_title = 'NO Resnet34 CAM Label'

if os.path.exists(SegMADe):
    y_seg_SegMADe = np.load(SegMADe)
    y_seg_SegMADe[y_seg_SegMADe > 0.5] = 1
    y_seg_SegMADe[y_seg_SegMADe <= 0.5] = 0
    SegMADe_title = 'SegMADe Label'
else:
    y_seg_SegMADe = np.zeros_like(y_seg_true)
    SegMADe_title = 'NO SegMADe Label'


def calc_dice(y_true_flat, y_pred_flat):
    intersection = np.sum(y_pred_flat * y_true_flat)
    smooth = 0.0000001
    dice = (2. * intersection + smooth) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + smooth)
    return dice


def get_edges(label):
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


def plot_on_ax(ppg, label, ax, title='', color='g', overlay=False, label2=None, color2='y'):
    ax.margins(x=0, y=0)
    ax.plot(ppg)
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


def plot_on_ax_pulse_tm(ppg, label, ax, title='', color='g', overlay=False, label2=None, color2='y'):
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
    ax.margins(x=0, y=0)
    ax.plot(ppg)

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


def vis_one(ppg, idx):
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, ncols=1, figsize=(15, 7))
    plot_on_ax(ppg, y_seg_true[idx], title='Human Label', overlay=False, ax=ax1)
    plot_on_ax(ppg, y_seg_SegMADe[idx],
               title=SegMADe_title + ' DICE: {}'.format(round(calc_dice(y_seg_true[idx], y_seg_SegMADe[idx]), 4)),
               overlay=False, ax=ax2)
    plot_on_ax(ppg, y_seg_cnnslider[idx],
               title=cnnslider_title + ' DICE: {}'.format(round(calc_dice(y_seg_true[idx], y_seg_cnnslider[idx]), 4)),
               overlay=False, ax=ax4)
    plot_on_ax_pulse_tm(ppg, y_seg_pulse_tm[idx],
                        title=pulsetm_title + ' DICE: {}'.format(
                            round(calc_dice(y_seg_true[idx], y_seg_pulse_tm[idx]), 4)),
                        overlay=False, ax=ax3)
    plot_on_ax(ppg, y_seg_res34_cam[idx],
               title=res34_cam_title + ' DICE: {}'.format(round(calc_dice(y_seg_true[idx], y_seg_res34_cam[idx]), 4)),
               overlay=False, ax=ax5)
    plot_on_ax(ppg, y_seg_res34_shap[idx],
               title=res34_shap_title + ' DICE: {}'.format(round(calc_dice(y_seg_true[idx], y_seg_res34_shap[idx]), 4)),
               overlay=False, ax=ax6)
    plt.tight_layout()
    plt.savefig('visualize_all/{}/{}.jpg'.format(TESTSET_NAME, idx))
    plt.close()
    plt.close('all')


pool_args = []
for idx, ppg in enumerate(X_test):
    pool_args.append([ppg, idx])

pool = Pool()
pool.starmap(vis_one, pool_args)
pool.terminate()
