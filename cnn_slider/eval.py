import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
from sklearn.metrics import classification_report, confusion_matrix
from model import *
from pathlib import Path
from tqdm import tqdm
from visualizer import *
from multiprocessing import pool

np.random.seed(1)
tf.random.set_seed(1)


def model_eval(
    fidx,
    working_dir,
    threshold,
    seg_length,
    prob_thresh,
    plot_limit,
    TESTSET_NAME,
):
    """
    Evaluation function for CNN sliding window method
    :param fidx: fold index range(0, 10)
    :type fidx: int
    :param working_dir: location/ directory to store fold evaluation results
    :type working_dir: str
    :param threshold: range(0.0, 1.0, 0.1), > threshold -> artifact, <= threshold -> clean
    :type threshold: float
    :param seg_length: sliding window length
    :type seg_length: int
    :param prob_thresh: range(0, 11, 1), probability threshold to calculate ROC
    :type prob_thresh: float
    :param plot_limit: number of plots to generate (set to 0 if no plots are needed or to reduce processing time)
    :type plot_limit: int
    :param TESTSET_NAME: test set name, make sure to have this named folder in parent 'data/' directory
    :type TESTSET_NAME: str
    :return: TPR, FPR, DICE score
    :rtype: float, float, float
    """

    plot_path = working_dir+'/plots/plot_{}_{}/'.format(threshold, prob_thresh)
    check_mkdir(plot_path)

    # loading data
    data_dir = str(Path(os.getcwd()).parent) + '/data/{}/'.format(TESTSET_NAME)
    X_test = np.load(data_dir+'/processed_dataset/scaled_ppgs.npy')
    y_seg_trues = np.load(data_dir+'/processed_dataset/seg_labels.npy')
    X_val_segs_threshold = np.load('data/{}_windows/X_val_segs_{}.npy'.format(TESTSET_NAME, threshold))
    y_slide_true_threshold = np.load('data/{}_windows/y_slide_true_{}.npy'.format(TESTSET_NAME, threshold))
    slices_threshold = np.load('data/{}_windows/slices_{}.npy'.format(TESTSET_NAME, threshold))
    y_raw_slide_preds = np.load('results/{}/thresh_{}_seg_length_{}/{}/y_raw_slide_preds_{}.npy'
                                .format(TESTSET_NAME, threshold, seg_length, fidx, threshold))

    # collect predictions, convert to segmentation
    y_seg_preds = []
    y_slide_preds = []
    y_slide_trues = []
    for idx, X_val_segs in enumerate(X_val_segs_threshold):
        y_slide_true = y_slide_true_threshold[idx]
        slices = slices_threshold[idx]
        y_slide_pred = y_raw_slide_preds[idx]
        y_slide_pred[y_slide_pred > prob_thresh] = 1
        y_slide_pred[y_slide_pred <= prob_thresh] = 0
        y_slide_pred = y_slide_pred.astype(np.int8)
        y_slide_preds.append(y_slide_pred)
        y_slide_trues.append(y_slide_true)

        y_seg_pred = process_sliding_results(y_slide_pred, y_seg_trues[idx], slices, prob_thresh)
        y_seg_preds.append(y_seg_pred)

    y_seg_preds = np.asarray(y_seg_preds)

    np.save(working_dir+'/y_seg_preds/y_pred_{}_{}.npy'.format(threshold, prob_thresh), y_seg_preds)

    y_pred_flat = y_seg_preds.flatten().astype(np.int8)
    y_true_flat = y_seg_trues.flatten().astype(np.int8)

    TPR, FPR = calc_TPR_FPR(y_true_flat, y_pred_flat)

    intersection = np.sum(y_pred_flat * y_true_flat)
    smooth = 0.0000001
    dice = (2. * intersection + smooth) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + smooth)

    n_transitions = 0
    for pred in y_seg_preds:
        n_transitions += len(get_edges(pred.flatten()))
    n_transitions /= y_seg_preds.shape[0]

    # draw visualizations
    if plot_limit > 0 and prob_thresh == 0.5:
        pbar = tqdm(X_test[:plot_limit, :], leave=False)
        for idx, cam in enumerate(pbar):
            ppg = X_test[idx]
            plot_result(ppg.flatten(), y_seg_trues[idx].flatten(), y_seg_preds[idx].flatten(),
                        save_path=plot_path+'/idx{}_probthresh{}.jpg'.format(idx, prob_thresh), show=False,
                        plot_prob=False)
            plt.clf()
            plt.close('all')

    return TPR, FPR, dice, n_transitions, prob_thresh
