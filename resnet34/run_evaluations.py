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

RUNSHAP = True

if __name__ == '__main__':
    TESTSET_NAME = sys.argv[1]
    plot_limiter = int(sys.argv[2])

    ##################################################################################################################
    # '''Preprocessing'''
    # # preprocess data
    # # convert segmentation labels to classification label
    # # resample signal length for resnet34
    # print('Preprocessing')
    # pbar1 = tqdm(range(10))
    # for fidx in pbar1:
    #     pbar1.set_description('Running fold {}'.format(fidx))
    #     pbar2 = tqdm(range(0, 11), leave=False)
    #     for thresh in pbar2:
    #         thresh = thresh / 10
    #         pbar2.set_description('Running threshold {}'.format(thresh))
    #         process_data(fidx, thresh, TESTSET_NAME, process_test_only=True)
    
    # ###################################################################################################################
    '''Postprocessing'''
    # run post_process_SHAP and post_process_CAM to calculate SHAP and CAM
    if RUNSHAP:
        os.system('python post_process_SHAP.py {}'.format(TESTSET_NAME))
    os.system('python post_process_CAM.py {}'.format(TESTSET_NAME))

    # ###################################################################################################################
    '''Segmentation Evaluation'''
    # evaluate segmentation
    print('Evaluating Segmentation')
    pbar3 = tqdm(range(0, 10))
    for class_threshold in pbar3:
        class_threshold = class_threshold / 10
        pbar3.set_description('Running threshold {}'.format(class_threshold))
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
            ''
            prob_thresh_dices_cam = []
            prob_thresh_dices_shap = []
            prob_thresh_n_transitions_cam = []
            prob_thresh_n_transitions_shap = []

            working_dir = 'results/{}/thresh_{}/{}/'.format(TESTSET_NAME, class_threshold, fidx)
            # utilize multiprocessing for evaluation
            pool_args = []
            for prob_thresh in range(11):
                prob_thresh = prob_thresh / 10
                pool_args.append([fidx, working_dir, class_threshold, prob_thresh, plot_limiter, TESTSET_NAME, RUNSHAP])
            pool = Pool()
            ret = pool.starmap(evaluate, pool_args)
            pool.terminate()
            for TPR_cam, FPR_cam, dice_cam, n_transition_cam, TPR_shap, FPR_shap, dice_shap, n_transition_shap, \
                        fidx_, threshold_, prob_thresh_ in ret:
                TPRs_cam.append(TPR_cam)
                FPRs_cam.append(FPR_cam)
                TPRs_shap.append(TPR_shap)
                FPRs_shap.append(FPR_shap)
                prob_thresh_dices_cam.append(dice_cam)
                prob_thresh_dices_shap.append(dice_shap)
                prob_thresh_n_transitions_cam.append(n_transition_cam)
                prob_thresh_n_transitions_shap.append(n_transition_shap)

            dices_cam.append(prob_thresh_dices_cam[0])
            n_transitions_cam.append(prob_thresh_n_transitions_cam[0])
            np.save(working_dir + '/Resnet34_TPRs_cam.npy', np.asarray(TPRs_cam))
            np.save(working_dir + '/Resnet34_FPRs_cam.npy', np.asarray(FPRs_cam))
            if RUNSHAP:
                dices_shap.append(prob_thresh_dices_shap[0])
                n_transitions_shap.append(prob_thresh_n_transitions_shap[0])
                np.save(working_dir + '/Resnet34_TPRs_shap.npy', np.asarray(TPRs_shap))
                np.save(working_dir + '/Resnet34_FPRs_shap.npy', np.asarray(FPRs_shap))

        dices_cam = np.asarray(dices_cam)
        n_transitions_cam = np.asarray(n_transitions_cam)
        check_mkdir('results/{}/segmentation_reports_CAM/'.format(TESTSET_NAME))

        # write dice, # of transitions to segmentation report files for both SHAP and CAM
        overall_report_file_cam = \
            open('results/{}/segmentation_reports_CAM/thresh_{}.txt'.format(TESTSET_NAME, class_threshold), 'w+')
        print('Dices: ', file=overall_report_file_cam)
        print(dices_cam, file=overall_report_file_cam)
        print('{} +- {}'.format(np.mean(dices_cam), np.std(dices_cam)), file=overall_report_file_cam)
        print('n_transitions: ', file=overall_report_file_cam)
        print(n_transitions_cam, file=overall_report_file_cam)
        print('{} +- {}'.format(np.mean(n_transitions_cam), np.std(n_transitions_cam)),
              file=overall_report_file_cam)
        overall_report_file_cam.close()

        if RUNSHAP:
            dices_shap = np.asarray(dices_shap)
            n_transitions_shap = np.asarray(n_transitions_shap)
            check_mkdir('results/{}/segmentation_reports_SHAP/'.format(TESTSET_NAME))

            overall_report_file_shap = \
                open('results/{}/segmentation_reports_SHAP/thresh_{}.txt'.format(TESTSET_NAME, class_threshold), 'w+')
            print('Dices: ', file=overall_report_file_shap)
            print(dices_shap, file=overall_report_file_shap)
            print('{} +- {}'.format(np.mean(dices_shap), np.std(dices_shap)), file=overall_report_file_shap)
            print('n_transitions: ', file=overall_report_file_shap)
            print(n_transitions_shap, file=overall_report_file_shap)
            print('{} +- {}'.format(np.mean(n_transitions_shap), np.std(n_transitions_shap)),
                  file=overall_report_file_shap)
            overall_report_file_shap.close()

    ###################################################################################################################
    '''Classification evaluation'''
    # evaluate classification results, calculate AUC via sklearn
    check_mkdir('results/{}/classification_reports/'.format(TESTSET_NAME))
    print('Evaluating Classification')

    class_threshold = 0.0
    aucs = []
    f1s = []
    pbar7 = tqdm(range(10), leave=False)
    for fidx in pbar7:
        pbar7.set_description('Running fold {}'.format(fidx))
        model_dir = 'model/thresh_{}/{}/'.format(class_threshold, fidx)
        tl_res34 = tf.keras.models.load_model(model_dir + '/model')
        tl_res34.load_weights(model_dir + '/{}_res34.h5'.format(class_threshold))
        X_test = np.load('data/{}/X_test_{}.npy'.format(TESTSET_NAME, class_threshold))
        y_preds = tl_res34.predict(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)))
        y_class_test = np.load('data/{}/y_class_test_{}.npy'.format(TESTSET_NAME, class_threshold))
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_class_test[:, 1].flatten(), y_preds[:, 1].flatten())
        f1 = metrics.f1_score(y_class_test[:, 1], np.argmax(y_preds, axis=1), average='macro')
        f1s.append(f1)
        auc = metrics.auc(fpr_keras, tpr_keras)
        aucs.append(auc)

    aucs = np.asarray(aucs)
    f1s = np.asarray(f1s)
    overall_report_file_auc = open(
        'results/{}/classification_report.txt'.format(TESTSET_NAME), 'w+')
    print(aucs, file=overall_report_file_auc)
    print('{} +- {}'.format(np.mean(aucs), np.std(aucs)), file=overall_report_file_auc)
    print('f1 score:', file=overall_report_file_auc)
    print(f1s, file=overall_report_file_auc)
    print('{} +- {}'.format(np.mean(f1s), np.std(f1s)), file=overall_report_file_auc)
    overall_report_file_auc.close()
