import numpy as np
import sqlite3
import math
from tqdm import tqdm
from scipy import signal
import io
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


DATASET_NAME = 'WESAD_all'
SAVE_FOLDER_NAME = 'WESAD_all'
# DATA_AMOUNT_LIMIT = 1000
DATA_AMOUNT_LIMIT = None

if not os.path.isdir('data/' + SAVE_FOLDER_NAME+'/'):
    os.mkdir('data/' + SAVE_FOLDER_NAME+'/')
if not os.path.isdir('data/' + SAVE_FOLDER_NAME + '/processed_dataset/'):
    os.mkdir('data/' + SAVE_FOLDER_NAME + '/processed_dataset/')


def ppg_process(ppg):
    fs = 64
    low_end = 0.9 / (fs / 2)
    high_end = 5 / (fs / 2)
    filter_order = 2

    sos = signal.butter(filter_order, [low_end, high_end], btype='bandpass', output='sos')
    filtered_ppg = signal.sosfilt(sos, ppg)

    ppg_norm = (filtered_ppg - min(filtered_ppg)) / (max(filtered_ppg) - min(filtered_ppg))
    return ppg_norm


def minmax_scale_ppg(ppg):
    ppg_norm = (ppg - min(ppg)) / (max(ppg) - min(ppg))
    return ppg_norm


def scale_ppg(ppg):
    fs = 64
    low_end = 0.9 / (fs / 2)
    high_end = 5 / (fs / 2)
    filter_order = 2

    sos = signal.butter(filter_order, [low_end, high_end], btype='bandpass', output='sos')
    filtered_ppg = signal.sosfilt(sos, ppg)

    ppg_norm = (filtered_ppg - min(filtered_ppg)) / (max(filtered_ppg) - min(filtered_ppg))
    return ppg_norm


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

data = sqlite3.connect('dataset/{}.db'.format(DATASET_NAME), detect_types=sqlite3.PARSE_DECLTYPES)
data_c = data.cursor()

anno = sqlite3.connect('dataset/{}_annotation_results.db'.format(DATASET_NAME), detect_types=sqlite3.PARSE_DECLTYPES)
anno_c = anno.cursor()

anno_c.execute('SELECT MAX(idx) FROM annotation_results LIMIT 1')
if DATA_AMOUNT_LIMIT:
    data_amount = DATA_AMOUNT_LIMIT
else:
    data_amount = anno_c.fetchone()[0]
print('Total {} data entry'.format(data_amount))

anno_dict = dict()
anno_bar = tqdm(range(0, data_amount+1))
for idx in anno_bar:
    anno_bar.set_description('{}'.format(idx))
    anno_c.execute("SELECT * FROM annotation_results WHERE idx={}".format(idx))
    rows = anno_c.fetchall()
    if rows is None:
        print('{} has no annotation'.format(idx))
        continue
    anno_dict[idx] = []
    for row in rows:
        idx, start, end = row
        if not start == -999 or end == -999:
            start = start / ((1 / 64) * 1000)
            end = end / ((1 / 64) * 1000)
            anno_dict[idx].append([start, end])
        else:
            print('{} skipped'.format(idx))

seg_label_dict = dict()

artifact_cnt = 0
total_cnt = 0

anno_dict_bar = tqdm(anno_dict.items())
for idx, annos in anno_dict_bar:
    anno_dict_bar.set_description('{}'.format(idx))
    seg_label_base = np.zeros((1920,))
    for start, end in annos:
        start = math.floor(start)
        end = math.ceil(end)
        seg_label_base[start: end] = 1

    artifact_cnt += np.sum(seg_label_base)
    total_cnt += seg_label_base.shape[0]

    seg_label_dict[idx] = seg_label_base

    u, cnts = np.unique(seg_label_base, return_counts=True)

print('Artifact percentage {}, artifact {}, non_artifact {}, total {}'.
      format(artifact_cnt / total_cnt, artifact_cnt, total_cnt - artifact_cnt, total_cnt))
anno.close()

ppgs = []
seg_labels = []
ppg_bar = tqdm(range(data_amount))
for idx in ppg_bar:
    data_c.execute("SELECT * FROM dataset WHERE idx={}".format(idx))
    data_row = data_c.fetchone()
    ecg, acc0, acc1, acc2, ppg, act, weight, gender, age, height, skin, sport = \
        np.frombuffer(data_row[1], dtype=np.float64), np.frombuffer(data_row[2], dtype=np.float64), \
        np.frombuffer(data_row[3], dtype=np.float64), np.frombuffer(data_row[4], dtype=np.float64), \
        np.frombuffer(data_row[5], dtype=np.float64), np.frombuffer(data_row[6], dtype=np.float64), \
        data_row[7], 'male' if data_row[8] == 0 else 'female', \
        data_row[9], data_row[10], data_row[11], data_row[12]

    # ppg = minmax_scale_ppg(ppg)
    # ppg = scale_ppg(ppg)
    ppgs.append(ppg)
    seg_labels.append(seg_label_dict[idx])

ppgs = np.asarray(ppgs)
seg_labels = np.asarray(seg_labels)

print(ppgs.shape, seg_labels.shape)

data.close()

np.save('data/' + '{}/processed_dataset/scaled_ppgs.npy'.format(SAVE_FOLDER_NAME), ppgs)
np.save('data/' + '{}/processed_dataset/seg_labels.npy'.format(SAVE_FOLDER_NAME), seg_labels)
