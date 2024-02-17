import numpy as np
from numpy import dstack
from pandas import read_csv
import pickle
import os
from sklearn.model_selection import train_test_split
from tsai.data.preparation import SlidingWindow

"""
https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset

18 classes, 51 subjects, sampling rate 20 Hz, each activity lasts for 3 minutes.
Follow https://www.sciencedirect.com/science/article/abs/pii/S0893608022004270, we use raw phone accelerator modality.
And set window_size and stride as 200. 
"""


root = '/home/qzz/Datasets/MTS-datasets/WISDM/raw/phone/accel/'
label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                 'M': 12, 'O': 13, 'P': 14, 'Q': 15, 'R': 16, 'S': 17}

window_len_wisdm, stride = 200, 200
# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None)
    return dataframe.values


def apply_sliding_window(record, cls, window_len, stride):
    sliding_windows, _ = SlidingWindow(window_len=window_len, stride=stride, seq_first=True,
                                       pad_remainder=True, padding_value=0, add_padding_feature=False
                                       )(record)
    sliding_windows = np.transpose(sliding_windows, (0, 2, 1))
    windows_labels = np.ones(sliding_windows.shape[0], dtype=float) * cls

    return sliding_windows, windows_labels


def load_dataset():
    data = []
    labels = []

    for sub in range(1600, 1651):
        filename = 'data_{}_accel_phone.txt'.format(sub)
        file = load_file(root+filename)
        file[:, -1] = np.array(list(map(lambda x: float(x[:-1]), file[:, -1])))  # Remove ; and turn str into float
        accel_data = file[:, -3:].astype(np.float64).transpose(0, 1)
        activity = file[:, 1]
        activity = np.array(list(map(lambda x: label_mapping[x], activity)))

        for i in range(0, 18):
            idx = np.where(activity == i)[0]
            data_i = accel_data[idx]
            sliding_windows, windows_labels = apply_sliding_window(data_i, i, window_len_wisdm, stride)
            data.append(sliding_windows)
            labels.append(windows_labels)

    data = np.concatenate(data)  # (19 * n_sub * 60, 125, 45)
    labels = np.concatenate(labels)  # (19 * n_sub * 60, )
    return data, labels


if __name__ == "__main__":
    # Save signals to file
    path = './saved/WISDM/'

    # Balanced train test split: all subjects
    trainX, trainy = load_dataset()
    trainX, testX, trainy, testy = train_test_split(trainX, trainy, test_size=0.25, random_state=0, stratify=trainy)

    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + 'x_train.pkl', 'wb') as f:
        pickle.dump(trainX, f)
    with open(path + 'x_test.pkl', 'wb') as f:
        pickle.dump(testX, f)
    with open(path + 'state_train.pkl', 'wb') as f:
        pickle.dump(trainy, f)
    with open(path + 'state_test.pkl', 'wb') as f:
        pickle.dump(testy, f)
