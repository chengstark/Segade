from train import *
from eval import *
from utils import *
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


TRAINING = False
EVALUATING = True

# TESTSET_NAME = 'TROIKA_channel_1'
TESTSET_NAME = 'PPG_DaLiA_test'


'''Training'''
if TRAINING:
    for fidx in range(10):
        model_train(
            fidx=fidx,
            plot_model_to_img=True,
            learning_rate=0.0005,
            n_epochs=200,
            batch_size=64
        )

'''Segmentation Evaluation'''
if EVALUATING:
    print('Evaluating Segmentation')
    dices = []
    n_transitions = []
    pbar1 = tqdm(range(10))
    for fidx in pbar1:
        pbar1.set_description('Running fold {}'.format(fidx))
        dice, n_transition = model_eval(fidx, TESTSET_NAME, filter_size=16, plot_limiter=5)
        dices.append(dice)
        n_transitions.append(n_transition)

    dices = np.asarray(dices)
    n_transitions = np.asarray(n_transitions)

    overall_report_file = open('results/{}/overall_eval_report.txt'.format(TESTSET_NAME), 'w+')
    print('Dices:', file=overall_report_file)
    print(dices, file=overall_report_file)
    print('{} +- {}'.format(np.mean(dices), np.std(dices)), file=overall_report_file)
    print('n_transitions:', file=overall_report_file)
    print(n_transitions, file=overall_report_file)
    print('{} +- {}'.format(np.mean(n_transitions), np.std(n_transitions)), file=overall_report_file)
    overall_report_file.close()

'''Classification Evaluation'''
if EVALUATING:
    print('Evaluating Classification')
    check_mkdir('results/{}/classification_reports/'.format(TESTSET_NAME))
    data_dir = str(Path(os.getcwd()).parent) + '/data/{}/'.format(TESTSET_NAME)
    y_true = np.load(data_dir + '/processed_dataset/seg_labels.npy')
    pbar2 = tqdm(range(0, 10))
    for class_threshold in pbar2:
        class_threshold = class_threshold / 10
        pbar2.set_description('Running classification threshold {}'.format(class_threshold))
        y_class_test = np.sum(y_true, axis=1) / y_true.shape[1]
        y_class_test[y_class_test > class_threshold] = 1
        y_class_test[y_class_test <= class_threshold] = 0
        aucs = []
        pbar3 = tqdm(range(10), leave=False)
        for fidx in pbar3:
            pbar3.set_description('Running fold {}'.format(fidx))
            working_dir = 'results/{}/{}/'.format(TESTSET_NAME, fidx)
            y_pred = np.load(working_dir + '/y_pred.npy')
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0

            TPRs = []
            FPRs = []
            pbar4 = tqdm(range(0, 11), leave=False)
            for prob_thresh in pbar4:
                prob_thresh = prob_thresh / 10
                pbar4.set_description('Running prob_thresh {}'.format(prob_thresh))

                y_class_pred = np.sum(y_pred, axis=1) / y_pred.shape[1]
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
        overall_report_file_auc = open(
            'results/{}/classification_reports/classthresh_{}.txt'.format(TESTSET_NAME, class_threshold), 'w+')
        print(aucs, file=overall_report_file_auc)
        print('{} +- {}'.format(np.mean(aucs), np.std(aucs)), file=overall_report_file_auc)
        overall_report_file_auc.close()















