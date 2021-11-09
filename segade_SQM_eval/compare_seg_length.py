import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import pickle as pkl
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from utils import *
from model import *
from scipy.signal import resample
from sklearn.metrics import roc_curve
from pathlib import Path
from visualizer import *


np.random.seed(1)
tf.random.set_seed(1)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


if __name__ == '__main__':
    data_dir = str(Path(os.getcwd()).parent) + '/data_folds/new_PPG_DaLiA_train/'
    tiny_seg_counts = []
    best_val_dices = []
    for fidx in range(10):
        model_dir = 'model_DICE/{}/'.format(fidx)
        # model_dir = 'model_BCE/{}/'.format(fidx)
        # model_dir = 'model_0L/{}/'.format(fidx)
        SegMADe = construct_SegMADe(filter_size=16)
        SegMADe.load_weights(model_dir + '/SegMADe_best.h5')
        X_val = np.load(data_dir + '/X_val_{}.npy'.format(fidx))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        y_vals = np.load(data_dir + '/y_seg_val_{}.npy'.format(fidx))
        y_vals = y_vals.reshape((y_vals.shape[0], y_vals.shape[1], 1))
        y_preds = SegMADe.predict(X_val).squeeze()

        y_preds[y_preds <= 0.5] = 0
        y_preds[y_preds > 0.5] = 1
        tiny_seg_count = 0
        for idx, y_pred in enumerate(y_preds):
            edges = get_edges(y_pred)

            # plot_result(X_test[idx].flatten(), true_label=y_pred, plot_true_only=True, show=False, save=True,
            #             save_path='tmp/{}.jpg'.format(idx), additional_title=' {}'.format(edges))

            for s, e in edges:
                # print(idx, e - s, s, e)

                if (e - s) < 48:
                    tiny_seg_count += 1
        tiny_seg_counts.append(tiny_seg_count / X_val.shape[0])
        intersection = np.sum(y_preds.flatten() * y_vals.flatten())
        smooth = 0.0000001
        dice = (2. * intersection + smooth) / (np.sum(y_vals.flatten()) + np.sum(y_preds.flatten()) + smooth)
        best_val_dices.append(dice)
        print(fidx, X_val.shape, tiny_seg_count, dice)

        # break



    print(sum(tiny_seg_counts) / len(tiny_seg_counts))
    print(sum(best_val_dices) / len(best_val_dices))
    best_val_dices = np.asarray(best_val_dices)
    tiny_seg_counts = np.asarray(tiny_seg_counts)
    print('{} +- {}'.format(np.mean(best_val_dices), np.std(best_val_dices)))
    print('{} +- {}'.format(np.mean(tiny_seg_counts), np.std(tiny_seg_counts)))





