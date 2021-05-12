import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage.filters import gaussian_filter1d


TESTSET_NAME = 'TROIKA_channel_1'


y_preds = []
dices = []
prob_thresh = 0.0
shaps = []
y_seg_preds_shaps = []
thresh = 0.0
y_seg_test = np.load('data/{}/y_seg_test_{}.npy'.format(TESTSET_NAME, thresh))

for fidx in range(10):
    # if fidx != 0: continuez
    shap = np.load('results/{}/thresh_{}/{}/shaps.npy'.format(TESTSET_NAME, thresh, fidx))
    shap = MinMaxScaler(feature_range=(0, 1)).fit_transform(shap)

    # smoothed_shap = []
    # for shap_ in shap:
    #     smoothed_shap.append(gaussian_filter1d(shap_, 7))
    # smoothed_shap = np.asarray(smoothed_shap)
    # shap = smoothed_shap

    shaps.append(shap)

    shap_seg = shap.copy()
    shap_seg[shap_seg > prob_thresh] = 1
    shap_seg[shap_seg <= prob_thresh] = 0
    y_seg_preds_shaps.append(shap_seg)
    print(shap_seg.shape, y_seg_test.shape)
    intersection = np.sum(shap_seg.flatten() * y_seg_test.flatten())
    smooth = 0.0000001
    dice = (2. * intersection + smooth) / (np.sum(y_seg_test.flatten()) + np.sum(shap_seg.flatten()) + smooth)
    dices.append(dice)

print(np.unique(np.equal(shaps[0], shaps[1])))
print(np.unique(np.equal(y_seg_preds_shaps[0], y_seg_preds_shaps[1])))
print(np.unique(np.equal(dices[0], dices[1])))
print(print(dices[0], dices[1]))