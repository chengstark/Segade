import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel("ERROR")
from sklearn.metrics import roc_curve
from train import *
from preprocessing import *
from postprocessing import *
from eval import *
from utils import *
from sklearn import metrics
import warnings
from multiprocessing import Pool
warnings.filterwarnings('ignore')

# TESTSET_NAME = 'PPG_DaLiA_test'
# TESTSET_NAME = 'TROIKA_channel_1'

if __name__ == '__main__':
    class_threshold = 0.0

    data_dir_train = str(Path(os.getcwd()).parent) + '/data/'
    y_seg_ori = np.load(data_dir_train + '/new_PPG_DaLiA_train/processed_dataset/seg_labels.npy')

    y_class_ori = np.sum(y_seg_ori, axis=1) / y_seg_ori.shape[1]
    y_class_ori[y_class_ori > class_threshold] = 1
    y_class_ori[y_class_ori <= class_threshold] = 0

    print(np.unique(y_class_ori, return_counts=True))

    aucs = []
    f1s = []
    pbar7 = tqdm(range(10), leave=False)
    for fidx in pbar7:
        pbar7.set_description('Running fold {}'.format(fidx))
        model_dir = 'model/thresh_{}/{}/'.format(class_threshold, fidx)
        tl_res34 = tf.keras.models.load_model(model_dir + '/model')
        tl_res34.load_weights(model_dir + '/{}_res34.h5'.format(class_threshold))
        # X_test = np.load('data/{}/X_test_{}.npy'.format(TESTSET_NAME, class_threshold))
        # y_preds = tl_res34.predict(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)))
        # y_class_test = np.load('data/{}/y_class_test_{}.npy'.format(TESTSET_NAME, class_threshold))

        X_val = np.load('data/X_val_{}_{}.npy'.format(fidx, class_threshold))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        y_class_val = np.load('data/y_class_val_{}_{}.npy'.format(fidx, class_threshold))
        y_preds = tl_res34.predict(X_val)

        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_class_val[:, 1].flatten(), y_preds[:, 1].flatten())
        f1 = metrics.f1_score(y_class_val[:, 1], np.argmax(y_preds, axis=1), average='macro')
        f1s.append(f1)
        auc = metrics.auc(fpr_keras, tpr_keras)
        aucs.append(auc)

    aucs = np.asarray(aucs)
    print('{} +- {}'.format(np.mean(aucs), np.std(aucs)))



