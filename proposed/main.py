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
