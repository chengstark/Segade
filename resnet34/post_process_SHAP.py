import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel("ERROR")
from keras import Model
from scipy.signal import resample
from visualizer import *
from utils import *
from scipy import interpolate
import shap

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


def post_process_SHAP(fidx, threshold, TESTSET_NAME):
    """
    Resnet34 post processing
    :param fidx: integer, fold index range(0, 10)
    :param threshold: float, range(0.0, 1.0, 0.1), > threshold -> artifact, <= threshold -> clean
    :param TESTSET_NAME: string, test set name, make sure to have this named folder in parent 'data/' directory
    :return: None
    """

    working_dir = 'results/{}/thresh_{}/{}/'.format(TESTSET_NAME, threshold, fidx)
    model_dir = 'model/thresh_{}/{}/'.format(threshold, fidx)

    check_mkdir('results/{}/'.format(TESTSET_NAME))
    check_mkdir('results/{}/thresh_{}/'.format(TESTSET_NAME, threshold))
    check_mkdir(working_dir)

    tl_res34 = tf.keras.models.load_model(model_dir+'/model')
    tl_res34.load_weights(model_dir+'/{}_res34.h5'.format(threshold))

    X_test = np.load('data/{}/X_test_{}.npy'.format(TESTSET_NAME, threshold))
    background = np.load('data/shap_background_{}.npy'.format(fidx))
    e = shap.DeepExplainer(tl_res34, background)
    shap_values = e.shap_values(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)))
    clean_shap = shap_values[0].squeeze()
    artifact_shap = shap_values[1].squeeze()
    clean_shap[clean_shap > 0] = 0
    artifact_shap[artifact_shap < 0] = 0
    shap_value = np.abs(clean_shap) + np.abs(artifact_shap)
    shap_downsampled = []
    for s in shap_value:
        shap_downsampled.append(resample(s, 1920))
    shap_downsampled = np.asarray(shap_downsampled)
    np.save(working_dir + '/shaps.npy', shap_downsampled)



if __name__ == '__main__':
    # TESTSET_NAME = 'TROIKA_channel_1'
    TESTSET_NAME = sys.argv[1]
    print('Post processing SHAP {}'.format(TESTSET_NAME))
    pbar1 = tqdm(range(8, 10))
    for thresh in pbar1:
        thresh = thresh / 10
        pbar1.set_description('Postprocessing threshold {}'.format(thresh))
        pbar2 = tqdm(range(0, 10), leave=False)
        for fidx in pbar2:
            pbar2.set_description('Postprocessing fold {}'.format(fidx, thresh))
            post_process_SHAP(fidx, thresh, TESTSET_NAME)

            tf.keras.backend.clear_session()
