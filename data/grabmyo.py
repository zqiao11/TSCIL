import scipy.io as scio
import numpy as np
import pickle
import os
from tsai.data.preparation import SlidingWindow
from sklearn.model_selection import train_test_split
from scipy.signal import resample


# 17 classes, 16 targeting motions and 1 for rest
# each recording is 5 seconds in duration, sampled at 2048 hz. So each has 10240 time steps.
# each activity performs 7 trials.
# N_samples per class: 43 * 7 * n_windows

physionet_root = '/home/qzz/Datasets/MTS-datasets/physionet.org/files/'
data_dir = physionet_root + 'grabmyo/'

DOWNSAMPLE = True
resample_length = 256 * 5  # downsample to 256 hz
window_len_grabmyo = 128  # each window is 1 sec
stride = 128  # no overlapping
GROUP = 'combined'

if GROUP == 'forearm':
    input_channels_grabmyo = 16
elif GROUP == 'wrist':
    input_channels_grabmyo = 12
elif GROUP == 'combined':
    input_channels_grabmyo = 28


N_sessions = 1
N_subjects = 43
N_classes = 16
N_trails = 7
# N_train_trails = 5


def apply_sliding_window(record, cls, window_len, stride):
    sliding_windows, _ = SlidingWindow(window_len=window_len, stride=stride, seq_first=True,
                                       pad_remainder=True, padding_value=0, add_padding_feature=False
                                       )(record)
    sliding_windows = np.transpose(sliding_windows, (0, 2, 1))
    windows_labels = np.ones(sliding_windows.shape[0], dtype=float) * cls

    return sliding_windows, windows_labels


def extract_samples_with_sliding_windows(window_len, stride, group='forearm', resampling=True):
    train_data, test_data = list(), list()
    train_labels, test_labels = list(), list()

    for session in range(1, N_sessions+1):

        for sub in range(1, N_subjects+1):
            filepath = data_dir + 'session{}_participant{}.mat'.format(session, sub)
            mat = scio.loadmat(filepath)  # Dictionary

            if group == 'forearm':
                collections = mat['DATA_FOREARM']  #  7 * 17, trials * activities
            elif group == 'wrist':
                collections = mat['DATA_WRIST']  # 7 * 17,
            elif group == 'combined':
                collections_forearm = mat['DATA_FOREARM']
                collections_wrist = mat['DATA_WRIST']
            else:
                raise ValueError("Wrong sensor group is given")

            ###################### Train-test split on all trials ######################
            for trail in range(0, N_trails):
                for cls in range(0, N_classes):  # Discard the last class 'rest'
                    if group in ['forearm', 'wrist']:
                        record = collections[trail][cls]  # 10240 time steps, 5 sec of 2048 hz
                        record = np.nan_to_num(record)
                    else:
                        record = np.concatenate((collections_forearm[trail][cls],collections_wrist[trail][cls]), axis=1)
                        record = np.nan_to_num(record)
                    if resampling:
                        record = resample(record, resample_length)

                    sliding_windows, windows_labels = apply_sliding_window(record, cls, window_len, stride)

                    train_data.append(sliding_windows)
                    train_labels.append(windows_labels)

    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels)
    return train_data, train_labels


if __name__ == "__main__":
    trainX, trainy = extract_samples_with_sliding_windows(window_len_grabmyo, stride, group=GROUP, resampling=DOWNSAMPLE)
    trainX, testX, trainy, testy = train_test_split(trainX, trainy, test_size=0.25, random_state=0, stratify=trainy)

    ## Save signals to file
    path = './saved/GRABMyo/'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + '/x_train.pkl', 'wb') as f:
        pickle.dump(trainX, f)  # trainX tasks around 500 MB
    with open(path + '/x_test.pkl', 'wb') as f:
        pickle.dump(testX, f)
    with open(path + '/state_train.pkl', 'wb') as f:
        pickle.dump(trainy, f)
    with open(path + '/state_test.pkl', 'wb') as f:
        pickle.dump(testy, f)