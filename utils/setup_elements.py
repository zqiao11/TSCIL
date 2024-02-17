# -*- coding: UTF-8 -*-
from data.grabmyo import window_len_grabmyo, input_channels_grabmyo
from data.ninapro import window_len_ninapro
from data.wisdm import window_len_wisdm
import os

content_root = os.path.abspath('.')

data_path = {
    'har': content_root + '/data/saved/HAR_inertial/',
    'uwave': content_root + '/data/saved/UWave/',
    'dailysports': content_root + '/data/saved/DailySports/',
    'grabmyo': content_root + '/data/saved/GRABMyo/',
    'ninapro': content_root + '/data/saved/Ninapro/',
    'wisdm': content_root + '/data/saved/WISDM/',
    'sines': content_root + '/data/saved/sines/'
    }

input_size_match = {
    'har': [128, 9],
    'uwave': [315, 3],
    'dailysports': [125, 45],
    'grabmyo': [window_len_grabmyo, input_channels_grabmyo],
    'ninapro': [window_len_ninapro, 12],
    'wisdm': [window_len_wisdm, 3],
    'sines': [128, 3]
    }

# jitter for Inversion
jitter = {
    'har': 10,
    'uwave': 10,
    'dailysports': 10,
    'grabmyo': 10,
    'ninapro': 10,
    'wisdm': 10,
          }

# Number of training sample per class
n_smp_per_cls = {
    'har': 1225,
    'uwave': 112,
    'dailysports': 360,
    'grabmyo': 2058,
    'wisdm': 1000,  # A approximate value, average value is 1074
    'ninapro': None,
    'sines': 500,
     }

n_classes = {
    'har': 6,
    'uwave': 8,
    'dailysports': 18,
    'grabmyo': 16,
    'ninapro': 16,
    'wisdm': 18,
    'sines': 4
    }

# Number of tasks in the entire task sequence
n_tasks = {
    'har': 3,
    'uwave': 4,
    'dailysports': 9,
    'grabmyo': 8,
    'ninapro': 8,
    'wisdm': 9,
    'sines':2
    }

# Number of tasks for validation
n_tasks_val = {
    'har': 3,
    'uwave': 4,
    'dailysports': 3,
    'grabmyo': 3,
    'ninapro': 3,
    'wisdm': 3,
    'sines':2
    }

# Number of tasks for experiment
n_tasks_exp = {k: v if k in ['har', 'uwave', 'sines'] else int(n_tasks[k] - v) for (k, v) in n_tasks_val.items()}

n_classes_per_task = {i: int(n_classes[i] / n_tasks[i]) for i in n_tasks.keys()}

preset_orders = {
    'har': list(range(6)),
    'uwave': [0, 2, 1, 3, 4, 6, 5, 7],
    'dailysports': list(range(18)),
    'grabmyo': list(range(16)),
    'ninapro': list(range(40)),
    'wisdm': list(range(18)),
    }


n_subjects = {'har': 21, 'dailysports': 8}


def get_num_classes(args):
    if args.stream_split == 'all':
        return n_classes[args.data]
    elif args.stream_split == 'val':
        return n_classes_per_task[args.data] * n_tasks_val[args.data]
    else:
        return n_classes_per_task[args.data] * n_tasks_exp[args.data]


def get_num_tasks(args):
    if args.stream_split == 'all':
        return n_tasks[args.data]
    elif args.stream_split == 'val':
        return n_tasks_val[args.data]
    else:
        return n_tasks_exp[args.data]


def get_buffer_size(args):
    n_exemplar_per_cls = int(args.mem_budget * n_smp_per_cls[args.data])
    n_exemplar_per_task = n_exemplar_per_cls * n_classes_per_task[args.data]
    if args.stream_split == 'all':
        mem_size = n_exemplar_per_task * n_tasks[args.data]
    elif args.stream_split == 'val':
        mem_size = n_exemplar_per_task * n_tasks_val[args.data]
    else:
        mem_size = n_exemplar_per_task * n_tasks_exp[args.data]

    return mem_size

