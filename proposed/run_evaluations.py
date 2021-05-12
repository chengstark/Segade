from train import *
from eval import *
from utils import *
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    TESTSET_NAME = sys.argv[1]
    plot_limiter = int(sys.argv[2])

    ###################################################################################################################
    '''Segmentation Evaluation'''
    print('Evaluating Segmentation')
    dices = []
    n_transitions = []
    pbar1 = tqdm(range(10))
    for fidx in pbar1:
        pbar1.set_description('Running fold {}'.format(fidx))
        dice, n_transition = model_eval(fidx, TESTSET_NAME, filter_size=16, plot_limiter=plot_limiter)
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

    ###################################################################################################################
    '''Classification Evaluation'''
    print('Evaluating Classification')
    check_mkdir('results/{}/classification_reports/'.format(TESTSET_NAME))
    data_dir = str(Path(os.getcwd()).parent) + '/data/{}/'.format(TESTSET_NAME)
    y_true = np.load(data_dir + '/processed_dataset/seg_labels.npy')
    X_test = np.load(data_dir + '/processed_dataset/scaled_ppgs.npy')
    class_threshold = 0.0
    y_class_test = np.sum(y_true, axis=1) / y_true.shape[1]
    y_class_test[y_class_test > class_threshold] = 1
    y_class_test[y_class_test <= class_threshold] = 0
    aucs = []
    f1s = []
    pbar3 = tqdm(range(10), leave=False)
    for fidx in pbar3:
        pbar3.set_description('Running fold {}'.format(fidx))
        working_dir = 'results/{}/{}/'.format(TESTSET_NAME, fidx)

        TPRs = []
        FPRs = []
        pbar4 = tqdm(range(0, 11), leave=False)
        for prob_thresh in pbar4:
            prob_thresh = prob_thresh / 10.0

            pbar4.set_description('Running prob_thresh {}'.format(prob_thresh))
            y_pred = np.load(working_dir + '/y_pred.npy')
            y_pred[y_pred > prob_thresh] = 1
            y_pred[y_pred <= prob_thresh] = 0
            y_class_pred = np.sum(y_pred, axis=1) / y_pred.shape[1]

            y_class_pred[y_class_pred > class_threshold] = 1
            y_class_pred[y_class_pred <= class_threshold] = 0
            TPR, FPR = calc_TPR_FPR(y_class_test.flatten(), y_class_pred.flatten())
            TPRs.append(TPR)
            FPRs.append(FPR)

            # cm = confusion_matrix(y_class_test, y_class_pred, normalize='true')
            # plot_confusion_matrix(cm, index=[0, 1], columns=[0, 1])
            # plt.savefig('cms/cm_{}_{}.jpg'.format(fidx, prob_thresh))
            # plt.clf()
            # plt.close('all')

            # if prob_thresh == 0.5:
            #     for idx, row in enumerate(X_test):
            #         y_pred2 = np.load(working_dir + '/y_pred.npy')[idx]
            #         y_true2 = np.load(data_dir + '/processed_dataset/seg_labels.npy')[idx]
            #         if y_class_test[idx] != y_class_pred[idx]:
            #             y_pred2[y_pred2 > prob_thresh] = 1
            #             y_pred2[y_pred2 <= prob_thresh] = 0
            #             plot_result(row, y_true2, y_pred2, show=False, save=True, save_path='TROIKA_wrong_classifications/_{}_{}.jpg'.format(fidx, idx))

        TPRs = np.asarray(TPRs + [1.0, 0.0])
        FPRs = np.asarray(FPRs + [1.0, 0.0])
        sorted_TPRs, sorted_FPRs = sort_TPRs_FPRs(TPRs, FPRs)
        print('FPRs', sorted_FPRs)
        print('TPRs', sorted_TPRs)
        auc = metrics.auc(sorted_FPRs, sorted_TPRs)
        aucs.append(auc)

        y_pred = np.load(working_dir + '/y_pred.npy')
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        y_class_pred = np.sum(y_pred, axis=1) / y_pred.shape[1]

        y_class_pred[y_class_pred > class_threshold] = 1
        y_class_pred[y_class_pred <= class_threshold] = 0
        f1 = metrics.f1_score(y_class_test, y_class_pred, average='macro')
        f1s.append(f1)

    aucs = np.asarray(aucs)
    overall_report_file_auc = open(
        'results/{}/classification_report.txt'.format(TESTSET_NAME, class_threshold), 'w+')
    print(aucs, file=overall_report_file_auc)
    print('{} +- {}'.format(np.mean(aucs), np.std(aucs)), file=overall_report_file_auc)
    print('f1 score:', file=overall_report_file_auc)
    print(f1s, file=overall_report_file_auc)
    print('{} +- {}'.format(np.mean(f1s), np.std(f1s)), file=overall_report_file_auc)
    overall_report_file_auc.close()

