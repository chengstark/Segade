from utils import *
from postprocessing import *
from eval import *
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

TRAINING = False
EVALUATING = True

# TESTSET_NAME = 'TROIKA_channel_1'
TESTSET_NAME = 'PPG_DaLiA_test'

'''Create templates'''
if TRAINING:
    for fidx in range(10):
        create_template(10, fidx)

'''Postprocessing'''
if EVALUATING:
    print('Postprocessing')
    pbar1 = tqdm(range(10))
    for fidx in pbar1:
        pbar1.set_description('Running fold {}'.format(fidx))
        pbar2 = tqdm(range(0, 21), leave=False)
        for dtw_thresh in pbar2:
            pbar2.set_description('Running dtw_thresh {}'.format(dtw_thresh))
            post_process(dtw_thresh, fidx, TESTSET_NAME)

'''Segmentation Evaluation'''
if EVALUATING:
    print('Evaluating Segmentation')
    check_mkdir('results/{}/segmentation_reports/'.format(TESTSET_NAME))
    dices_dict = dict()
    n_transitions_dict = dict()
    for dtw_thresh in range(0, 21):
        dices_dict[dtw_thresh] = []
        n_transitions_dict[dtw_thresh] = []

    pbar3 = tqdm(range(10))
    for fidx in pbar3:
        pbar3.set_description('Running fold {}'.format(fidx))
        TPRs = []
        FPRs = []

        pbar4 = tqdm(range(0, 20), leave=False)
        for dtw_thresh in pbar4:
            pbar4.set_description('Running dtw_thresh {}'.format(dtw_thresh))
            TPR, FPR, dice, n_transition = template_eval(dtw_thresh, 0, fidx, TESTSET_NAME)
            TPRs.append(TPR)
            FPRs.append(FPR)

            dices_dict[dtw_thresh].append(dice)
            n_transitions_dict[dtw_thresh].append(n_transition)

        TPRs = np.asarray(TPRs)
        FPRs = np.asarray(FPRs)

        np.save('results/{}/{}/template_match_TPRs.npy'.format(TESTSET_NAME, fidx), TPRs)
        np.save('results/{}/{}/template_match_FPRs.npy'.format(TESTSET_NAME, fidx), FPRs)

    for dtw_thresh in dices_dict.keys():
        dices = np.asarray(dices_dict[dtw_thresh])
        n_transitions = np.asarray(n_transitions_dict[dtw_thresh])
        overall_report_file = open('results/{}/segmentation_reports/dtw_thresh_{}.txt'.format(TESTSET_NAME, dtw_thresh), 'w+')
        print('Dices:', file=overall_report_file)
        print(dices, file=overall_report_file)
        print('{} +- {}'.format(np.mean(dices), np.std(dices)), file=overall_report_file)
        print('n_transitions:', file=overall_report_file)
        print(n_transitions, file=overall_report_file)
        print('{} +- {}'.format(np.mean(n_transitions), np.std(n_transitions)), file=overall_report_file)
        overall_report_file.close()


'''Evaluate Classification'''
if EVALUATING:
    print('Evaluating Classification')
    check_mkdir('results/{}/classification_reports/'.format(TESTSET_NAME))

    data_dir = str(Path(os.getcwd()).parent) + '/data/{}/'.format(TESTSET_NAME)
    y_true = np.load(data_dir + '/processed_dataset/seg_labels.npy')

    pbar5 = tqdm(range(0, 10))
    for class_threshold in pbar5:
        class_threshold = class_threshold / 10
        pbar5.set_description('Running classification threshold {}'.format(class_threshold))
        y_class_test = np.sum(y_true, axis=1) / y_true.shape[1]
        y_class_test[y_class_test > class_threshold] = 1
        y_class_test[y_class_test <= class_threshold] = 0

        pbar6 = tqdm(range(0, 21), leave=False)
        for dtw_thresh in pbar6:
            pbar6.set_description('Running dtw_thresh {}'.format(dtw_thresh))
            aucs = []
            pbar7 = tqdm(range(10), leave=False)
            for fidx in pbar7:
                pbar7.set_description('Running fold {}'.format(fidx))
                working_dir = 'results/{}/{}/dtw_thresh_{}/'.format(TESTSET_NAME, fidx, dtw_thresh)
                y_seg_preds = np.load(working_dir + '/y_pred_{}.npy'.format(dtw_thresh, dtw_thresh))
                TPRs = []
                FPRs = []
                pbar8 = tqdm(range(0, 11), leave=False)
                for prob_thresh in pbar8:
                    prob_thresh = prob_thresh / 10
                    pbar8.set_description('Running prob_thresh {}'.format(prob_thresh))
                    y_class_pred = np.sum(y_seg_preds, axis=1) / y_seg_preds.shape[1]
                    y_class_pred[y_class_pred > prob_thresh] = 1
                    y_class_pred[y_class_pred <= prob_thresh] = 0
                    TPR, FPR = calc_TPR_FPR(y_class_test.flatten(), y_class_pred.flatten())
                    TPRs.append(TPR)
                    FPRs.append(FPR)

                TPRs = np.asarray(TPRs + [1])
                FPRs = np.asarray(FPRs + [1])
                sorted_TPRs, sorted_FPRs = sort_TPRs_FPRs(TPRs, FPRs)
                auc = metrics.auc(sorted_FPRs, sorted_TPRs)
                aucs.append(auc)
            aucs = np.asarray(aucs)

            overall_report_file_auc = \
                open('results/{}/classification_reports/classthresh_{}_dtwthresh_{}_report.txt'
                     .format(TESTSET_NAME, class_threshold, dtw_thresh), 'w+')
            print(aucs, file=overall_report_file_auc)
            print('{} +- {}'.format(np.mean(aucs), np.std(aucs)), file=overall_report_file_auc)
            overall_report_file_auc.close()




