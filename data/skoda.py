import numpy as np
from numpy import dstack
from pandas import read_csv, read_table
import pickle
import os
from sklearn.model_selection import train_test_split
from tsai.data.preparation import SlidingWindow

"""
http://har-dataset.org/doku.php?id=wiki:dataset

10 classes, 1 subjects, 70 instances per cls, recording is about 3h long. sampling rate 100 Hz, each activity lasts for  minutes.
https://github.com/andreasKyratzis/PAMAP2-Physical-Activity-Monitoring-Data-Analysis-and-ML/blob/master/pamap2.ipynb

Missing values.
Unbalanced dataset.
"""

