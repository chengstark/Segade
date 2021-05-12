from eval import *
from utils import *
from train import *
from eval import *
from postprocessing import *
from sklearn import metrics
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve


train_amount = 5000
val_amount = int(train_amount * (0.2 / 0.75))
epoch = 200
seg_length = 64 * 3
interval = 64
assert 1920 % seg_length == 0

if __name__ == '__main__':
    threshold = 0.0
    aucs = []
    for fidx in range(10):
        X_train = np.load('data/X_train_{}_{}.npy'.format(threshold, fidx))
        X_val = np.load('data/X_val_{}_{}.npy'.format(threshold, fidx))
        y_train = np.load('data/y_train_{}_{}.npy'.format(threshold, fidx))
        y_val = np.load('data/y_val_{}_{}.npy'.format(threshold, fidx))

        print(np.unique(y_train, return_counts=True))
        model_dir = 'model/thresh_{}_seg_length_{}/{}/'.format(threshold, seg_length, fidx)
        model = get_model(seg_length)
        model.load_weights(model_dir + '/slider_best_{}.h5'.format(threshold))

        y_preds = model.predict(X_val)
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_val.flatten(), y_preds.flatten())

        auc = metrics.auc(fpr_keras, tpr_keras)
        aucs.append(auc)

    aucs = np.asarray(aucs)
    print('{} +- {}'.format(np.mean(aucs), np.std(aucs)))



