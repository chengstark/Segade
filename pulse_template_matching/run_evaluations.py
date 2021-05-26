from utils import *
from postprocessing import *
from eval import *
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')



if __name__ == '__main__':
    TESTSET_NAME = sys.argv[1]
    plot_limiter = int(sys.argv[2])

    ###################################################################################################################
    # '''Postprocessing'''
    print('Postprocessing')
    pbar1 = tqdm(range(10))
    for fidx in pbar1:
        pbar1.set_description('Running fold {}'.format(fidx))
        pbar2 = tqdm(range(0, 11), leave=False)
        for dtw_thresh in pbar2:
            pbar2.set_description('Running dtw_thresh {}'.format(dtw_thresh))
            post_process(dtw_thresh, fidx, TESTSET_NAME)
    
    ###################################################################################################################
    '''Segmentation Evaluation'''
    print('Evaluating Segmentation')
    check_mkdir('results/{}/segmentation_reports/'.format(TESTSET_NAME))
    dices_dict = dict()
    n_transitions_dict = dict()
    for dtw_thresh in range(0, 11):
        dices_dict[dtw_thresh] = []
        n_transitions_dict[dtw_thresh] = []
    
    pbar3 = tqdm(range(10))
    for fidx in pbar3:
        pbar3.set_description('Running fold {}'.format(fidx))
        TPRs = []
        FPRs = []
    
        pbar4 = tqdm(range(0, 11), leave=False)
        for dtw_thresh in pbar4:
            pbar4.set_description('Running dtw_thresh {}'.format(dtw_thresh))
            TPR, FPR, dice, n_transition = template_eval(dtw_thresh, plot_limiter, fidx, TESTSET_NAME)
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
        overall_report_file = open(
            'results/{}/segmentation_reports/dtw_thresh_{}.txt'.format(TESTSET_NAME, dtw_thresh), 'w+')
        print('Dices:', file=overall_report_file)
        print(dices, file=overall_report_file)
        print('{} +- {}'.format(np.mean(dices), np.std(dices)), file=overall_report_file)
        print('n_transitions:', file=overall_report_file)
        print(n_transitions, file=overall_report_file)
        print('{} +- {}'.format(np.mean(n_transitions), np.std(n_transitions)), file=overall_report_file)
        overall_report_file.close()

