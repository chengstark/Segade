import numpy as np
from utils import *
import os
from multiprocessing import Pool
import time


def post_process(dtw_threshold, fidx, TESTSET_NAME):
    """
    Post process the template matching method, make predictions
    :param dtw_threshold: integer, dynamic time warping threshold
    :param fidx: integer, fold index range(0, 10)
    :param TESTSET_NAME: string, test set name, make sure to have this named folder in parent 'data/' directory
    :return: None
    """

    data_dir = str(Path(os.getcwd()).parent) + '/data/{}/'.format(TESTSET_NAME)
    X_test = np.load(data_dir+'/processed_dataset/scaled_ppgs.npy')
    y_seg_trues = np.load(data_dir+'/processed_dataset/seg_labels.npy')

    working_dir = 'results/{}/{}/dtw_thresh_{}/'.format(TESTSET_NAME, fidx, dtw_threshold)

    check_mkdir('results/{}'.format(TESTSET_NAME))
    check_mkdir('results/{}/{}/'.format(TESTSET_NAME, fidx))
    check_mkdir(working_dir)

    pool_args = []
    for row in X_test:
        pool_args.append([row, dtw_threshold, fidx])
    pool = Pool(processes=8)
    y_seg_preds = pool.starmap(make_predictions, pool_args)
    pool.terminate()

    y_seg_preds = np.asarray(y_seg_preds)
    np.save(working_dir+'/y_pred_{}.npy'.format(dtw_threshold), y_seg_preds)
    np.save(working_dir+'/y_true_{}.npy'.format(dtw_threshold), y_seg_trues)


