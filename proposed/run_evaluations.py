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


