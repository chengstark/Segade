from eval import *
from utils import *
from train import *
from eval import *
from preprocessing import *
from postprocessing import *
from sklearn import metrics
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

TRAINING = True
EVALUATING = True

# TESTSET_NAME = 'TROIKA_channel_1'
TESTSET_NAME = 'PPG_DaLiA_test'


'''Params'''
train_amount = 5000
val_amount = int(train_amount * (0.2 / 0.75))
epoch = 200
seg_length = 64 * 3
interval = 64
assert 1920 % seg_length == 0


'''preprocessing training set'''
if TRAINING:
    for threshold in tqdm(range(0, 10)):
        threshold = threshold / 10
        preprocess(train_amount, threshold, seg_length)


'''Training'''
if TRAINING:
    for threshold in range(0, 10):
        threshold = threshold / 10
        for fidx in range(10):
            model_train(train_amount, val_amount, threshold, seg_length, epoch, fidx)

'''Postprocessing'''
if EVALUATING:
    post_process(TESTSET_NAME, seg_length, interval)

'''Segmentation Evaluation'''
if EVALUATING:

    print('Evaluating Segmentation')
    data_dir = str(Path(os.getcwd()).parent) + '/data/{}/'.format(TESTSET_NAME)
    y_true = np.load(data_dir + '/processed_dataset/seg_labels.npy')

    pbar2 = tqdm(range(10))
    for threshold in pbar2:
        threshold = threshold / 10
        pbar2.set_description('Running threshold {}'.format(threshold))

        aucs = []
        dices = []
        n_transitions = []

        pbar3 = tqdm(range(10), leave=False)
        for fidx in pbar3:
            pbar3.set_description('Running fold {}'.format(fidx))

            TPRs = []
            FPRs = []

            working_dir = 'results/{}/thresh_{}_seg_length_{}/{}/'.format(TESTSET_NAME, threshold, seg_length, fidx)
            check_mkdir('results/{}/'.format(TESTSET_NAME))
            check_mkdir('results/{}/thresh_{}_seg_length_{}/'.format(TESTSET_NAME, threshold, seg_length))
            check_mkdir(working_dir)

            report_file = open(working_dir + '/eval_report.txt', 'w+')

            pbar4 = tqdm(range(0, 11), leave=False)
            for prob_thresh in pbar4:
                prob_thresh = prob_thresh / 10
                pbar4.set_description('Running prob_thresh {}'.format(prob_thresh))

                TPR, FPR, dice, n_transition = \
                    model_eval(
                        fidx,
                        working_dir=working_dir,
                        threshold=threshold,
                        seg_length=seg_length,
                        prob_thresh=prob_thresh,
                        plot_limit=0,
                        TESTSET_NAME=TESTSET_NAME
                    )

                TPRs.append(TPR)
                FPRs.append(FPR)

                if prob_thresh == 0.5:
                    dices.append(dice)
                    n_transitions.append(n_transition)

            report_file.close()

            np.save(working_dir+'/CNNSlider_TPRs.npy', np.asarray(TPRs))
            np.save(working_dir+'/CNNSlider_FPRs.npy', np.asarray(FPRs))

        dices = np.asarray(dices)

        check_mkdir('results/{}/segmentation_reports/'.format(TESTSET_NAME))
        overall_report_file = open('results/{}/segmentation_reports/thresh_{}_seg_length_{}.txt'
                                   .format(TESTSET_NAME, threshold, seg_length), 'w+')
        print('Dices: ', file=overall_report_file)
        print(dices, file=overall_report_file)
        print('{} +- {}'.format(np.mean(dices), np.std(dices)), file=overall_report_file)
        print('n_transitions: ', file=overall_report_file)
        print(n_transitions, file=overall_report_file)
        print('{} +- {}'.format(np.mean(n_transitions), np.std(n_transitions)), file=overall_report_file)
        overall_report_file.close()


'''Classification evaluation'''
if EVALUATING:
    print('Evaluating Classification')
    check_mkdir('results/{}/classification_reports/'.format(TESTSET_NAME))
    data_dir = str(Path(os.getcwd()).parent) + '/data/{}/'.format(TESTSET_NAME)
    y_true = np.load(data_dir + '/processed_dataset/seg_labels.npy')
    pbar5 = tqdm(range(10))
    for class_threshold in pbar5:
        class_threshold = class_threshold / 10
        pbar5.set_description('Running classification threshold {}'.format(class_threshold))

        y_class_test = np.sum(y_true, axis=1) / y_true.shape[1]
        y_class_test[y_class_test > class_threshold] = 1
        y_class_test[y_class_test <= class_threshold] = 0

        pbar6 = tqdm(range(10), leave=False)
        for seg_threshold in pbar6:
            seg_threshold = seg_threshold / 10
            pbar6.set_description('Running segmentation threshold {}'.format(seg_threshold))

            aucs = []
            pbar7 = tqdm(range(10), leave=False)
            for fidx in pbar7:
                pbar7.set_description('Running fold {}'.format(fidx))

                TPRs = []
                FPRs = []
                working_dir = 'results/{}/thresh_{}_seg_length_{}/{}/'.format(TESTSET_NAME, seg_threshold, seg_length, fidx)
                y_seg_preds = np.load(working_dir + '/y_seg_preds/y_pred_{}_{}.npy'.format(seg_threshold, 0.5))

                pbar8 = tqdm(range(11), leave=False)
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
                open('results/{}/classification_reports/classthresh_{}_segthresh_{}.txt'
                     .format(TESTSET_NAME, class_threshold, seg_threshold), 'w+')
            print(aucs, file=overall_report_file_auc)
            print('{} +- {}'.format(np.mean(aucs), np.std(aucs)), file=overall_report_file_auc)
            overall_report_file_auc.close()





