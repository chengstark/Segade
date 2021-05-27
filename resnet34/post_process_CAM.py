import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from visualizer import *
from utils import *
from scipy import interpolate

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel("ERROR")
from keras import Model


def post_process_CAM(fidx, threshold, TESTSET_NAME):
    """
    Resnet34 post processing for Grad CAM
    :param fidx: fold index range(0, 10)
    :type fidx: int
    :param threshold: range(0.0, 1.0, 0.1), > threshold -> artifact, <= threshold -> clean
    :type threshold: float
    :param TESTSET_NAME: test set name
    :type TESTSET_NAME: str
    :return: None
    :rtype: None
    """

    working_dir = 'results/{}/thresh_{}/{}/'.format(TESTSET_NAME, threshold, fidx)
    model_dir = 'model/thresh_{}/{}/'.format(threshold, fidx)

    check_mkdir('results/{}/'.format(TESTSET_NAME))
    check_mkdir('results/{}/thresh_{}/'.format(TESTSET_NAME, threshold))
    check_mkdir(working_dir)

    tl_res34 = tf.keras.models.load_model(model_dir+'/model')
    tl_res34.load_weights(model_dir+'/{}_res34.h5'.format(threshold))

    X_test = np.load('data/{}/X_test_{}.npy'.format(TESTSET_NAME, threshold))

    if X_test.shape[0] > 400:
        X_test_chunks = []
        for i in range(1, X_test.shape[0] // 400 + 2):
            X_test_chunks.append(X_test[(i-1)*400:i*400, :])
            # print((i-1)*400, i*400, X_test[(i-1)*400:i*400, :].shape)
    else:
        X_test_chunks = [X_test]
    raw_cams_chunks = []
    y_class_pred_chunks = []
    for X_test_chunk in X_test_chunks:
        X_test_reshaped = X_test_chunk.reshape((X_test_chunk.shape[0], X_test_chunk.shape[1], 1))

        predict = tl_res34.predict(X_test_reshaped)
        target_class = np.argmax(predict, axis=1)
        y_class_pred_chunks.append(target_class)
        last_conv_layer = tl_res34.layers[-5]
        grad_model = Model([tl_res34.input], [last_conv_layer.output, tl_res34.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(X_test_reshaped)
            loss = tf.gather_nd(predictions,
                                np.hstack((np.arange(predictions.shape[0]).reshape((predictions.shape[0], 1)),
                                           target_class.reshape((target_class.shape[0], 1)))))

        outputs = conv_outputs
        grads = tape.gradient(loss, conv_outputs)
        guided_grads = tf.cast(outputs > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
        weights = tf.reduce_mean(guided_grads, axis=1)
        raw_cams = np.ones((outputs.shape[0], outputs.shape[1])).astype(np.float32)
        for i in range(weights.shape[0]):
            weight = weights[i]
            output = outputs[i]
            p = np.dot(output, weight)
            raw_cams[i] += p

        raw_cams_chunks.append(raw_cams)

    raw_cams = np.concatenate((raw_cams_chunks))
    np.save(working_dir+'/cams.npy', raw_cams)

    y_class_preds = np.concatenate((y_class_pred_chunks))
    np.save(working_dir+'/y_class_preds.npy', y_class_preds)


if __name__ == '__main__':
    # TESTSET_NAME = 'TROIKA_channel_1'
    TESTSET_NAME = sys.argv[1]
    print('Post processing CAM {}'.format(TESTSET_NAME))
    pbar1 = tqdm(range(8, 10))
    for thresh in pbar1:
        thresh = thresh / 10
        pbar1.set_description('Postprocessing threshold {}'.format(thresh))
        pbar2 = tqdm(range(0, 10), leave=False)
        for fidx in pbar2:
            pbar2.set_description('Postprocessing fold {}'.format(fidx, thresh))
            post_process_CAM(fidx, thresh, TESTSET_NAME)


