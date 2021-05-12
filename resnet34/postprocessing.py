from visualizer import *
from utils import *


def post_process(fidx, threshold, TESTSET_NAME, shap_explainer=False):
    """
    Resnet34 post processing
    :param fidx: integer, fold index range(0, 10)
    :param threshold: float, range(0.0, 1.0, 0.1), > threshold -> artifact, <= threshold -> clean
    :param TESTSET_NAME: string, test set name, make sure to have this named folder in parent 'data/' directory
    :param shap_explainer: #TBD
    :return: None
    """

    import os
    from scipy import interpolate

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    import tensorflow as tf

    if shap_explainer:
        tf.compat.v1.disable_v2_behavior()
        tf.compat.v1.disable_eager_execution()
    else:
        tf.compat.v1.enable_v2_behavior()
        tf.compat.v1.enable_eager_execution()

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.get_logger().setLevel("ERROR")
    from keras import Model
    from scipy.signal import resample
    import shap

    working_dir = 'results/{}/thresh_{}/{}/'.format(TESTSET_NAME, threshold, fidx)
    model_dir = 'model/thresh_{}/{}/'.format(threshold, fidx)

    check_mkdir('results/{}/'.format(TESTSET_NAME))
    check_mkdir('results/{}/thresh_{}/'.format(TESTSET_NAME, threshold))
    check_mkdir(working_dir)

    tl_res34 = tf.keras.models.load_model(model_dir+'/model')
    tl_res34.load_weights(model_dir+'/{}_res34.h5'.format(threshold))

    X_test = np.load('data/{}/X_test_{}.npy'.format(TESTSET_NAME, threshold))
    X_train = np.load('data/X_train_{}_{}.npy'.format(fidx, threshold))

    if shap_explainer:
        # y_class_train = np.load('data/y_class_train_{}_{}.npy'.format(fidx, threshold))
        # background = X_train[np.where(y_class_train[:, 1] == 0)[0]]
        background = np.load('data/shap_background_{}.npy'.format(fidx))
        e = shap.DeepExplainer(tl_res34, background)
        shap_values = e.shap_values(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)))
        shap_value = np.abs(shap_values[1].squeeze())
        shap_downsampled = []
        for s in shap_value:
            shap_downsampled.append(resample(s, 1920))
        shap_downsampled = np.asarray(shap_downsampled)
        np.save(working_dir + '/shaps.npy', shap_downsampled)

    else:
        if X_test.shape[0] > 400:
            X_test_chunks = [X_test[:400, :], X_test[400:, :]]
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

        if len(raw_cams_chunks) > 1:
            raw_cams = np.concatenate((raw_cams_chunks[0], raw_cams_chunks[1]))
        else:
            raw_cams = raw_cams_chunks[0]
        np.save(working_dir+'/cams.npy', raw_cams)

        if len(y_class_pred_chunks) > 1:
            y_class_preds = np.concatenate((y_class_pred_chunks[0], y_class_pred_chunks[1]))
        else:
            y_class_preds = y_class_pred_chunks[0]
        np.save(working_dir+'/y_class_preds.npy', y_class_preds)


