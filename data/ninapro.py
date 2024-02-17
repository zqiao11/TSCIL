import scipy.io as scio
import numpy as np
import pickle
import os
from tsai.data.preparation import SlidingWindow
from sklearn.model_selection import train_test_split
from scipy.signal import resample
from itertools import groupby
from operator import itemgetter


data_dir = '/home/qzz/Datasets/MTS-datasets/Ninapro/DB2/'

DOWNSAMPLE = True

downsample_rate = 10
window_len_ninapro = int(2000 / downsample_rate)  # 1 sec
stride = window_len_ninapro
# stride = int(window_len_ninapro / 2)  # no overlapping

N_exercises = 1  # 1 or 2
N_subjects = 5  # For now, we only use 5. 40 in original database.
N_classes = 16 if N_exercises == 1 else 40
N_trails = 6
# N_train_trails = 5

def apply_sliding_window(record, cls, window_len, stride, padding=False):
    sliding_windows, _ = SlidingWindow(window_len=window_len, stride=stride, seq_first=True,
                                       pad_remainder=padding, padding_value=0, add_padding_feature=False
                                       )(record)
    sliding_windows = np.transpose(sliding_windows, (0, 2, 1))
    windows_labels = np.ones(sliding_windows.shape[0], dtype=float) * cls

    return sliding_windows, windows_labels


def extract_samples_with_sliding_windows(window_len, stride, resampling=True):
    train_data, test_data = list(), list()
    train_labels, test_labels = list(), list()

    for sub in range(1, N_subjects+1):
        sub_dir = data_dir + 'DB2_s{}/'.format(sub)

        for exe in range(1, N_exercises+1):
            filepath = sub_dir + 'S{}_E{}_emg.mat'.format(sub, exe)
            mat = scio.loadmat(filepath)  # Dictionary
            emg_record = mat['emg']  # nparray (L, 12)
            label_record = np.squeeze(mat['restimulus'])  # (L,)
            (cls_start, cls_end) = (1, 16) if exe == 1 else (17, 40)

            for cls in range(cls_start, cls_end+1):  # 0 is rest, discard it.
                cls_inds = np.where(label_record == cls)[0]
                segment_idx = []  # Extract the start/end index of 6 repetitions
                for k, g in groupby(enumerate(cls_inds), lambda i_x: i_x[0] - i_x[1]):
                    consecutive_signal = list(map(itemgetter(1), g))
                    segment_idx.append((consecutive_signal[0], consecutive_signal[-1] + 1))
                assert len(segment_idx) == N_trails, "Error occurs during extracting repeats"

                for (start, end) in segment_idx:
                    repetition = emg_record[start: end]  # Each is around 5 sec with 2k HZ. Around 7k~15k time steps.
                    repetition = np.nan_to_num(repetition)

                    if resampling:
                        resample_length = int(repetition.shape[0] / downsample_rate)
                        repetition = resample(repetition, resample_length)

                    sliding_windows, windows_labels = apply_sliding_window(repetition, cls, window_len, stride)

                    train_data.append(sliding_windows)
                    train_labels.append(windows_labels)

    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels)
    return train_data, train_labels


if __name__ == "__main__":
    trainX, trainy = extract_samples_with_sliding_windows(window_len_ninapro, stride, resampling=True)
    trainX, testX, trainy, testy = train_test_split(trainX, trainy, test_size=0.25, random_state=0, stratify=trainy)

    # trainX, trainy, testX, testy = extract_samples_with_sliding_windows(window_len_wisdm=window_len_grabmyo, stride=stride)

    ## Save signals to file
    path = './saved/Ninapro/'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + '/x_train.pkl', 'wb') as f:
        pickle.dump(trainX, f)  # trainX tasks around 500 MB
    with open(path + '/x_test.pkl', 'wb') as f:
        pickle.dump(testX, f)
    with open(path + '/state_train.pkl', 'wb') as f:
        pickle.dump(trainy-1, f)
    with open(path + '/state_test.pkl', 'wb') as f:
        pickle.dump(testy-1, f)
