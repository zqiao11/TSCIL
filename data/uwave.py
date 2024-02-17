from scipy.io import arff
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split

path = '/home/qzz/Datasets/MTS-datasets/UEA/'

"""
http://www.timeseriesclassification.com/description.php?Dataset=UWaveGestureLibraryAll
"""


def load_uwave():
    """
    3 variables, fixed sequence length 315, 8 classes
    Training size 896, Test size 3582
    """
    uwx_path = path + 'UWaveGestureLibraryX/'
    uwy_path = path + 'UWaveGestureLibraryY/'
    uwz_path = path + 'UWaveGestureLibraryZ/'

    x_train = arff.loadarff(uwx_path + 'UWaveGestureLibraryX_TRAIN.arff')
    x_test  = arff.loadarff(uwx_path + 'UWaveGestureLibraryX_TEST.arff')
    y_train = arff.loadarff(uwy_path + 'UWaveGestureLibraryY_TRAIN.arff')
    y_test  = arff.loadarff(uwy_path + 'UWaveGestureLibraryY_TEST.arff')
    z_train = arff.loadarff(uwz_path + 'UWaveGestureLibraryZ_TRAIN.arff')
    z_test  = arff.loadarff(uwz_path + 'UWaveGestureLibraryZ_TEST.arff')

    dfx_train, dfy_train, dfz_train = pd.DataFrame(x_train[0]), pd.DataFrame(y_train[0]), pd.DataFrame(z_train[0])
    dfx_test, dfy_test, dfz_test = pd.DataFrame(x_test[0]), pd.DataFrame(y_test[0]), pd.DataFrame(z_test[0])

    x_train, y_train, z_train = dfx_train.to_numpy(), dfy_train.to_numpy(), dfz_train.to_numpy()
    x_test, y_test, z_test = dfx_test.to_numpy(), dfy_test.to_numpy(), dfz_test.to_numpy()

    label_train, label_test = x_train[:, -1].astype(int), x_test[:, -1].astype(int)

    data_train = np.stack([x_train[:, :-1], y_train[:, :-1], z_train[:, :-1]], axis=2)
    data_test  = np.stack([x_test[:, :-1], y_test[:, :-1], z_test[:, :-1]], axis=2)

    data_train, data_val, label_train, label_val = train_test_split(data_train, label_train, test_size=0.1, random_state=0)
    print(data_train.shape, label_train.shape)
    print(data_val.shape, label_val.shape)
    print(data_test.shape, label_test.shape)

    ## Save signals to file
    np_path = './saved/UWave'
    if not os.path.exists(np_path):
        os.mkdir(np_path)

    with open(np_path + '/x_train.pkl', 'wb') as f:
        pickle.dump(data_train.astype('float64'), f)
    with open(np_path + '/x_test.pkl', 'wb') as f:
        pickle.dump(data_test.astype('float64'), f)
    with open(np_path + '/x_val.pkl', 'wb') as f:
        pickle.dump(data_val.astype('float64'), f)

    with open(np_path + '/state_train.pkl', 'wb') as f:
        pickle.dump(label_train - 1, f)
    with open(np_path + '/state_test.pkl', 'wb') as f:
        pickle.dump(label_test - 1, f)
    with open(np_path + '/state_val.pkl', 'wb') as f:
        pickle.dump(label_val - 1, f)

if __name__ == "__main__":
    load_uwave()
