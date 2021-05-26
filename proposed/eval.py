import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
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


def model_eval(fidx, TESTSET_NAME, filter_size=16, plot_limiter=20):
    """
    UNet model evaluation
    :rtype: object
    :param fidx: integer, fold index range(0, 10)
    :param TESTSET_NAME: string, test set name, make sure to have this named folder in parent 'data/' directory
    :param filter_size: integer, UNet initial layer filter size
    :param plot_limiter: integer, number of plots to generate (set to 0 if no plots are needed or to reduce processing time)
    :return: float DICE score
    """
    working_dir = 'results/{}/{}/'.format(TESTSET_NAME, fidx)

    check_mkdir('results/{}/'.format(TESTSET_NAME))
    check_mkdir(working_dir)
    check_mkdir(working_dir+'/{}/'.format(TESTSET_NAME))
    check_mkdir(working_dir+'/plots/')

    report_file = open(working_dir + '/eval_report.txt', 'w+')

    data_dir = str(Path(os.getcwd()).parent) + '/data/{}/'.format(TESTSET_NAME)
    X_test = np.load(data_dir + '/processed_dataset/scaled_ppgs.npy')
    y_true = np.load(data_dir + '/processed_dataset/seg_labels.npy')

    model_dir = 'model/{}/'.format(fidx)
    unet = construct_unet(filter_size=filter_size)
    unet.load_weights(model_dir + '/unet_best.h5')
    y_preds_ = unet.predict(X_test).squeeze()
    y_preds = y_preds_.copy()
    np.save(working_dir+'/y_pred.npy', y_preds)

    y_true_flat = y_true.flatten().astype(np.int8)

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true_flat, y_preds.flatten())
    np.save(working_dir+'/UNet_TPRs.npy', tpr_keras)
    np.save(working_dir+'/UNet_FPRs.npy', fpr_keras)

    y_preds[y_preds <= 0.5] = 0
    y_preds[y_preds > 0.5] = 1

    y_pred_flat = y_preds.flatten().astype(np.int8)

    print(classification_report(y_true_flat, y_pred_flat, target_names=["0", "1"]), file=report_file)
    print('\n', file=report_file)

    intersection = np.sum(y_pred_flat * y_true_flat)
    smooth = 0.0000001
    dice = (2. * intersection + smooth) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + smooth)
    print('DICE: {}\n'.format(dice), file=report_file)
    report_file.close()

    n_transitions = 0
    for pred in y_preds:
        n_transitions += len(get_edges(pred.flatten()))
    n_transitions /= y_preds.shape[0]

    if plot_limiter > 0:
        pbar = tqdm(X_test[:plot_limiter, :], position=0, leave=False)

        for idx, ppg in enumerate(pbar):
            print(y_preds_[idx].flatten())
            plot_result(
                ppg, y_true[idx].flatten(), y_preds[idx].flatten(), raw_prediction=y_preds_[idx].flatten(), show=False, save=True,
                save_path=working_dir + 'plots/{}.jpg'.format(idx), plot_prob=True
            )
            idx += 1

    return dice, n_transitions


if __name__ == '__main__':
    for fidx in range(10):
        model_eval(fidx, 'new_PPG_DaLiA_test', plot_limiter=20)
        break
    # working_dir = 'results/{}/{}/'.format('new_PPG_DaLiA_test', 0)
    # p = np.load(working_dir+'/y_pred.npy')
    # np.save('sampelsx20.npy', p[:20])
    # print(p[:20].shape)
    # for i, row in enumerate(p[:20]):
    #     plt.plot(row)
    #     plt.savefig('tmp/{}.jpg'.format(i))
    #     plt.clf()

