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
warnings.filterwarnings('ignore')


TRAINING = False
EVALUATING = True

# TESTSET_NAME = 'TROIKA_channel_1'
TESTSET_NAME = 'PPG_DaLiA_test'

'''Preprocessing'''
print('Preprocessing')
pbar1 = tqdm(range(10))
for fidx in pbar1:
    pbar1.set_description('Running fold {}'.format(fidx))
    pbar2 = tqdm(range(0, 11), leave=False)
    for thresh in pbar2:
        thresh = thresh / 10
        pbar2.set_description('Running threshold {}'.format(thresh))
        process_data(fidx, thresh, TESTSET_NAME, process_test_only=EVALUATING)

# '''Training'''
# if TRAINING:
#     for thresh in range(0, 10):
#         thresh = thresh / 10
#         for fidx in range(10):
#             train_res34(fidx, thresh)
#
'''Postprocessing'''
if EVALUATING:
    os.system('python post_process_SHAP.py {}'.format(TESTSET_NAME))
    os.system('python post_process_CAM.py {}'.format(TESTSET_NAME))

'''Segmentation Evaluation'''
if EVALUATING:
    print('Evaluating Segmentation')
    pbar3 = tqdm(range(0, 10))
    for threshold in pbar3:
        threshold = threshold / 10
        pbar3.set_description('Running threshold {}'.format(threshold))

        dices_cam = []
        dices_shap = []
        n_transitions_cam = []
        n_transitions_shap = []

        pbar4 = tqdm(range(0, 10), leave=False)
        for fidx in pbar4:
            pbar4.set_description('Running fold {}'.format(fidx))

            TPRs_cam = []
            FPRs_cam = []
            TPRs_shap = []
            FPRs_shap = []
            prob_thresh_dices_cam = []
            prob_thresh_dices_shap = []
            prob_thresh_n_transitions_cam = []
            prob_thresh_n_transitions_shap = []

            working_dir = 'results/{}/thresh_{}/{}/'.format(TESTSET_NAME, threshold, fidx)
            report_file = open(working_dir + '/eval_report.txt', 'w+')

            pbar5 = tqdm(range(0, 11), leave=False)
            for prob_thresh in pbar5:
                prob_thresh = prob_thresh / 10
                pbar5.set_description('Running prob_thresh {}'.format(prob_thresh))

                s = time.time()
                TPR_cam, FPR_cam, dice_cam, n_transition_cam, TPR_shap, FPR_shap, dice_shap, n_transition_shap = \
                    evaluate(fidx, working_dir, report_file, threshold, prob_thresh, 0, TESTSET_NAME)
                TPRs_cam.append(TPR_cam)
                FPRs_cam.append(FPR_cam)
                TPRs_shap.append(TPR_shap)
                FPRs_shap.append(FPR_shap)
                prob_thresh_dices_cam.append(dice_cam)
                prob_thresh_dices_shap.append(dice_shap)
                prob_thresh_n_transitions_cam.append(n_transition_cam)
                prob_thresh_n_transitions_shap.append(n_transition_shap)

            dices_cam.append(max(prob_thresh_dices_cam))
            dices_shap.append(max(prob_thresh_dices_shap))
            n_transitions_cam.append(prob_thresh_n_transitions_cam[prob_thresh_dices_cam.index(max(prob_thresh_dices_cam))])
            n_transitions_shap.append(prob_thresh_n_transitions_shap[prob_thresh_dices_shap.index(max(prob_thresh_dices_shap))])


            report_file.close()

            np.save(working_dir + '/Resnet34_TPRs_cam.npy', np.asarray(TPRs_cam))
            np.save(working_dir + '/Resnet34_FPRs_cam.npy', np.asarray(FPRs_cam))
            np.save(working_dir + '/Resnet34_TPRs_shap.npy', np.asarray(TPRs_shap))
            np.save(working_dir + '/Resnet34_FPRs_shap.npy', np.asarray(FPRs_shap))

        dices_cam = np.asarray(dices_cam)
        dices_shap = np.asarray(dices_shap)
        n_transitions_cam = np.asarray(n_transitions_cam)
        n_transitions_shap = np.asarray(n_transitions_shap)
        check_mkdir('results/{}/segmentation_reports_CAM/'.format(TESTSET_NAME))
        check_mkdir('results/{}/segmentation_reports_SHAP/'.format(TESTSET_NAME))
        overall_report_file_cam = \
            open('results/{}/segmentation_reports_CAM/thresh_{}.txt'.format(TESTSET_NAME, threshold), 'w+')
        print('Dices: ', file=overall_report_file_cam)
        print(dices_cam, file=overall_report_file_cam)
        print('{} +- {}'.format(np.mean(dices_cam), np.std(dices_cam)), file=overall_report_file_cam)
        print('n_transitions: ', file=overall_report_file_cam)
        print(n_transitions_cam, file=overall_report_file_cam)
        print('{} +- {}'.format(np.mean(n_transitions_cam), np.std(n_transitions_cam)),
              file=overall_report_file_cam)
        overall_report_file_cam.close()

        overall_report_file_shap = \
            open('results/{}/segmentation_reports_SHAP/thresh_{}.txt'.format(TESTSET_NAME, threshold), 'w+')
        print('Dices: ', file=overall_report_file_shap)
        print(dices_shap, file=overall_report_file_shap)
        print('{} +- {}'.format(np.mean(dices_shap), np.std(dices_shap)), file=overall_report_file_shap)
        print('n_transitions: ', file=overall_report_file_shap)
        print(n_transitions_shap, file=overall_report_file_shap)
        print('{} +- {}'.format(np.mean(n_transitions_shap), np.std(n_transitions_shap)),
              file=overall_report_file_shap)
        overall_report_file_shap.close()


'''Classification evaluation'''
if EVALUATING:
    check_mkdir('results/{}/classification_reports/'.format(TESTSET_NAME))
    print('Evaluating Classification')

    pbar6 = tqdm(range(10))
    for threshold in pbar6:
        threshold = threshold / 10
        pbar6.set_description('Running threshold {}'.format(threshold))
        aucs = []
        pbar7 = tqdm(range(10), leave=False)
        for fidx in pbar7:
            pbar7.set_description('Running fold {}'.format(fidx))
            model_dir = 'model/thresh_{}/{}/'.format(threshold, fidx)
            tl_res34 = tf.keras.models.load_model(model_dir + '/model')
            tl_res34.load_weights(model_dir + '/{}_res34.h5'.format(threshold))
            X_test = np.load('data/{}/X_test_{}.npy'.format(TESTSET_NAME, threshold))
            y_preds = tl_res34.predict(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)))
            y_class_test = np.load('data/{}/y_class_test_{}.npy'.format(TESTSET_NAME, threshold))
            fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_class_test[:, 1].flatten(), y_preds[:, 1].flatten())
            auc = metrics.auc(fpr_keras, tpr_keras)
            aucs.append(auc)

        aucs = np.asarray(aucs)
        overall_report_file_auc = open(
            'results/{}/classification_reports/thresh_{}.txt'.format(TESTSET_NAME, threshold), 'w+')
        print(aucs, file=overall_report_file_auc)
        print('{} +- {}'.format(np.mean(aucs), np.std(aucs)), file=overall_report_file_auc)
        overall_report_file_auc.close()
