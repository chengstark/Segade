from utils import *
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from train import *


def post_process(TESTSET_NAME, seg_length, interval):
    data_dir = str(Path(os.getcwd()).parent) + '/data/{}/'.format(TESTSET_NAME)
    X_test = np.load(data_dir + '/processed_dataset/scaled_ppgs.npy')
    y_seg_trues = np.load(data_dir + '/processed_dataset/seg_labels.npy')
    print(X_test.shape, y_seg_trues.shape)
    check_mkdir('data/{}_windows/'.format(TESTSET_NAME))
    print('Generating sliding windows')
    for threshold in tqdm(range(10)):
        threshold = threshold / 10
        X_val_segs_threshold = []
        y_slide_true_threshold = []
        slices_threshold = []
        for idx, row in enumerate(X_test):
            X_val_segs, y_slide_true, slices = slide_window(row, y_seg_trues[idx], seg_length, interval, threshold)
            X_val_segs_threshold.append(X_val_segs)
            y_slide_true_threshold.append(y_slide_true)
            slices_threshold.append(slices)
        X_val_segs_threshold = np.asarray(X_val_segs_threshold)
        y_slide_true_threshold = np.asarray(y_slide_true_threshold)
        slices_threshold = np.asarray(slices_threshold)
        np.save('data/{}_windows/X_val_segs_{}.npy'.format(TESTSET_NAME, threshold), X_val_segs_threshold)
        np.save('data/{}_windows/y_slide_true_{}.npy'.format(TESTSET_NAME, threshold), y_slide_true_threshold)
        np.save('data/{}_windows/slices_{}.npy'.format(TESTSET_NAME, threshold), slices_threshold)

    print('Making predictions')
    pbar2 = tqdm(range(10))
    for threshold in pbar2:
        threshold = threshold / 10
        pbar2.set_description('Running threshold {}'.format(threshold))

        pbar3 = tqdm(range(10), leave=False)
        for fidx in pbar3:
            check_mkdir('results/{}/'.format(TESTSET_NAME))
            check_mkdir('results/{}/thresh_{}_seg_length_{}/'.format(TESTSET_NAME, threshold, seg_length))
            check_mkdir('results/{}/thresh_{}_seg_length_{}/{}/'.format(TESTSET_NAME, threshold, seg_length, fidx))
            pbar3.set_description('Running fold {}'.format(fidx))
            model_dir = 'model/thresh_{}_seg_length_{}/{}/'.format(threshold, seg_length, fidx)
            model = get_model(seg_length)
            model.load_weights(model_dir + '/slider_best_{}.h5'.format(threshold))
            X_val_segs_threshold = np.load('data/{}_windows/X_val_segs_{}.npy'.format(TESTSET_NAME, threshold))
            raw_preds = []
            for idx, X_val_segs in enumerate(X_val_segs_threshold):
                X_val_segs = X_val_segs.reshape(X_val_segs.shape[0], X_val_segs.shape[1], 1)
                y_slide_pred = model.predict(X_val_segs)
                raw_preds.append(y_slide_pred)
            raw_preds = np.asarray(raw_preds)
            np.save('results/{}/thresh_{}_seg_length_{}/{}/y_raw_slide_preds_{}.npy'
                    .format(TESTSET_NAME, threshold, seg_length, fidx, threshold), raw_preds)


if __name__ == '__main__':
    train_amount = 5000
    val_amount = int(train_amount * (0.2 / 0.75))
    epoch = 200
    seg_length = 64 * 3
    interval = 64
    assert 1920 % seg_length == 0

    TESTSET_NAME = 'PPG_DaLiA_test'
    post_process(TESTSET_NAME, seg_length, interval)

