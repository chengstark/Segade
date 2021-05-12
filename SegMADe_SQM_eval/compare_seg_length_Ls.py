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
    all_counts = []
    all_DICEs = []
    for L in range(7):
        tiny_seg_counts = []
        best_val_dices = []
        for fidx in range(10):
            # model_dir = 'model_dice/{}/'.format(fidx)
            # model_dir = 'model_bce/{}/'.format(fidx)
            # model_dir = 'model_nolength/{}/'.format(fidx)
            # model_dir = 'model_ac/{}/'.format(fidx)
            model_dir = 'model_{}L/{}/'.format(L, fidx)
            unet = construct_unet(filter_size=16)
            unet.load_weights(model_dir + 'unet_best.h5')
            X_vals = np.load(data_dir + '/X_val_{}.npy'.format(fidx))
            X_vals = X_vals.reshape((X_vals.shape[0], X_vals.shape[1], 1))
            y_vals = np.load(data_dir + '/y_seg_val_{}.npy'.format(fidx))
            y_vals = y_vals.reshape((y_vals.shape[0], y_vals.shape[1], 1))
            y_preds = unet.predict(X_vals).squeeze()
            # data_dir = str(Path(os.getcwd()).parent) + '/data/{}/'.format('PPG_DaLiA_test')
            # X_test = np.load(data_dir + '/processed_dataset/scaled_ppgs.npy')
            # y_true = np.load(data_dir + '/processed_dataset/seg_labels.npy')
            # y_preds = unet.predict(X_test).squeeze()

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
            tiny_seg_counts.append(tiny_seg_count / X_vals.shape[0])
            print(X_vals.shape, tiny_seg_count)
            intersection = np.sum(y_preds.flatten() * y_vals.flatten())
            smooth = 0.0000001
            dice = (2. * intersection + smooth) / (np.sum(y_vals.flatten()) + np.sum(y_preds.flatten()) + smooth)
            best_val_dices.append(dice)

            # break



        print(sum(tiny_seg_counts) / len(tiny_seg_counts))
        print(sum(best_val_dices) / len(best_val_dices))
        all_counts.append(sum(tiny_seg_counts) / len(tiny_seg_counts))
        all_DICEs.append(sum(best_val_dices) / len(best_val_dices))

    plt.margins(x=0, y=0)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    plt.plot(range(7), all_counts)
    plt.xlabel('$\lambda_{t}$')
    plt.ylabel('Signal Quality Measure')
    plt.title('$\lambda_{t}$ vs Signal Quality Measure')

    plt.plot(range(7), all_DICEs)

    plt.savefig('lambda_t vs Signal Quality Measure.jpg')
    plt.clf()



