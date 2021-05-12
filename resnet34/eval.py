import os
from scipy import interpolate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf

import keras.backend as K
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel("ERROR")
from model import resnet_34
from utils import *
from keras.layers import *
from keras import Model
import pickle as pkl
from scipy.signal import resample
from sklearn.metrics import accuracy_score
import shap
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from pathlib import Path
from visualizer import *
from multiprocessing import Pool
from vis import utils as vis_utils
from vis.visualization import visualize_cam
from keras import activations
import keras
import time
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage.filters import gaussian_filter1d


def evaluate(fidx, working_dir, threshold, prob_thresh, plot_limit, TESTSET_NAME, runshap):
    """
    Resnet34 evaluation with cams
    :param fidx: integer, fold index range(0, 10)
    :param working_dir: string, location/ directory to store fold evaluation results
    :param threshold: float, range(0.0, 1.0, 0.1), > threshold -> artifact, <= threshold -> clean
    :param prob_thresh: float, range(0, 11, 1), probability threshold to calculate ROC
    :param plot_limit: integer, number of plots to generate (set to 0 if no plots are needed or to reduce processing time)
    :param TESTSET_NAME: string, test set name, make sure to have this named folder in parent 'data/' directory
    :return: TPR, FPR, DICE score
    """

    def process_y_seg_preds(y_seg_test, y_seg_preds, probs, explainer, plot_limit):
        y_seg_preds_flat = y_seg_preds.flatten().astype(np.int8)
        y_seg_trues_flat = y_seg_test.flatten().astype(np.int8)

        # print(classification_report(y_seg_trues_flat, y_seg_preds_flat, target_names=["0", "1"]), file=report_file)
        # print('\n', file=report_file)

        intersection = np.sum(y_seg_preds_flat * y_seg_trues_flat)
        smooth = 0.0000001
        dice = (2. * intersection + smooth) / (np.sum(y_seg_trues_flat) + np.sum(y_seg_preds_flat) + smooth)
        # if explainer == 'shap':
        #     print('dice: ', fidx, threshold, prob_thresh, dice)
        TPR, FPR = calc_TPR_FPR(y_seg_trues_flat, y_seg_preds_flat)

        n_transitions = 0
        for pred in y_seg_preds:
            n_transitions += len(get_edges(pred.flatten()))
        n_transitions /= y_seg_preds.shape[0]

        if plot_limit > 0:
            check_mkdir(working_dir + '/plots_{}/'.format(explainer))
            check_mkdir(working_dir + '/plots_{}/{}/'.format(explainer, prob_thresh))
            pbar = tqdm(probs[:plot_limit], leave=False)
            for idx, prob in enumerate(pbar):
                ppg_1920 = X_test_1920[idx]

                plot_result(ppg_1920.flatten(), y_seg_test[idx].flatten(), y_seg_preds[idx].flatten(), prob=prob,
                            save_path=working_dir + '/plots_{}/{}/{}.jpg'.format(explainer, prob_thresh, idx), show=False,
                            plot_prob=True, title_addition=' {} Model Label {}'.format(np.argmax(y_class_test[idx]), y_class_preds[idx]),
                            explainer=explainer)
                plt.clf()
                plt.close('all')

        return TPR, FPR, dice, n_transitions

    y_class_test = np.load('data/{}/y_class_test_{}.npy'.format(TESTSET_NAME, threshold))
    y_seg_test = np.load('data/{}/y_seg_test_{}.npy'.format(TESTSET_NAME, threshold))

    y_class_preds = np.load(working_dir + '/y_class_preds.npy')
    raw_cams = np.load(working_dir + '/cams.npy')

    data_dir = str(Path(os.getcwd()).parent) + '/data/{}/'.format(TESTSET_NAME)
    X_test_1920 = np.load(data_dir + '/processed_dataset/scaled_ppgs.npy')

    raw_cams = (raw_cams - np.min(raw_cams)) / (np.max(raw_cams) - np.min(raw_cams))
    cams = []
    for raw_cam in raw_cams:
        cam = resample(raw_cam, 1920)
        cams.append(cam)
    cams = np.asarray(cams)

    '''Evaluate GRAD CAM'''
    y_seg_preds_cams = []
    for idx, cam in enumerate(cams):
        cam_seg = cam.copy()
        prediction = y_class_preds[idx]
        if prediction == 1:
            cam_seg[cam > prob_thresh] = 1
            cam_seg[cam <= prob_thresh] = 0
        else:
            cam_seg[:] = 0
        y_seg_preds_cams.append(cam_seg)
    y_seg_preds_cams = np.asarray(y_seg_preds_cams)
    np.save(working_dir+'y_seg_preds_{}_{}_{}_{}.npy'.format('cam', fidx, threshold, prob_thresh), y_seg_preds_cams)

    TPR_shap, FPR_shap, dice_shap, n_transitions_shap = None, None, None, None

    if runshap:
        shaps = np.load(working_dir + '/shaps.npy')
        '''Evaluate SHAP'''
        shaps = MinMaxScaler(feature_range=(0, 1)).fit_transform(shaps)
        smoothed_shaps = []
        for shap_ in shaps:
            smoothed_shaps.append(gaussian_filter1d(shap_, 7))
        shaps = np.asarray(smoothed_shaps)
        y_seg_preds_shaps = shaps.copy()
        y_seg_preds_shaps[y_seg_preds_shaps > prob_thresh] = 1
        y_seg_preds_shaps[y_seg_preds_shaps <= prob_thresh] = 0
        # for idx, shap_ in enumerate(smoothed_shaps):
        #     shap_seg = shap_.copy()
        #     shap_seg[shap_seg > prob_thresh] = 1
        #     shap_seg[shap_seg <= prob_thresh] = 0
        #     y_seg_preds_shaps.append(shap_seg)
        # y_seg_preds_shaps = np.asarray(y_seg_preds_shaps)
        np.save(working_dir + 'y_seg_preds_{}_{}_{}_{}.npy'.format('shap', fidx, threshold, prob_thresh),
                y_seg_preds_shaps)
        TPR_shap, FPR_shap, dice_shap, n_transitions_shap = process_y_seg_preds(y_seg_test, y_seg_preds_shaps,
                                                                                shaps, 'shap', plot_limit)

    TPR_cam, FPR_cam, dice_cam, n_transitions_cam = process_y_seg_preds(y_seg_test, y_seg_preds_cams, cams, 'cam', plot_limit)
    # print(fidx, threshold, prob_thresh, dice_shap)
    return TPR_cam, FPR_cam, dice_cam, n_transitions_cam, TPR_shap, FPR_shap, dice_shap, n_transitions_shap, fidx, threshold, prob_thresh




