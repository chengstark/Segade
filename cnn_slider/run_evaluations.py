from eval import *
from utils import *
from train import *
from eval import *
from postprocessing import *
from sklearn import metrics
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')


'''Params'''
train_amount = 5000
val_amount = int(train_amount * (0.2 / 0.75))
epoch = 200
seg_length = 64 * 3
interval = 64
assert 1920 % seg_length == 0


if __name__ == '__main__':
    TESTSET_NAME = sys.argv[1]
    plot_limiter = int(sys.argv[2])

    ###################################################################################################################
    '''Postprocessing test set'''
    # post process the test set results
    # prepare for the sliding windows, save to fields
    # make predictions on the test sets based on thresholds
    post_process(TESTSET_NAME, seg_length, interval)

    ###################################################################################################################
    '''Segmentation Evaluation'''
    # evaluate segmentation
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
            check_mkdir(working_dir + '/y_seg_preds/')
            check_mkdir(working_dir + '/plots/')

            # use multiprocessing to handle evaluations
            pool_args = []
            for prob_thresh in range(11):
                prob_thresh = prob_thresh / 10
                pool_args.append([fidx, working_dir, threshold, seg_length, interval, prob_thresh, plot_limiter, TESTSET_NAME])
            pool = Pool()
            ret = pool.starmap(model_eval, pool_args)
            pool.terminate()
            for TPR, FPR, dice, n_transition, prob_thresh in ret:
                TPRs.append(TPR)
                FPRs.append(FPR)
                if prob_thresh == 0.5:
                    dices.append(dice)
                    n_transitions.append(n_transition)

            np.save(working_dir + '/CNNSlider_TPRs.npy', np.asarray(TPRs))
            np.save(working_dir + '/CNNSlider_FPRs.npy', np.asarray(FPRs))

        dices = np.asarray(dices)

        # write dice, # of transitions to file
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
