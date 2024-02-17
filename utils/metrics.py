import numpy as np
from scipy.stats import sem
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def compute_performance(end_task_acc_arr):
    """
    Given test accuracy results from multiple runs saved in end_task_acc_arr,
    compute the average accuracy, forgetting, and task accuracies as well as their confidence intervals.

    :param end_task_acc_arr:        3D nd_arrays  (num_runs, num_tasks, num_tasks)
    :param task_ids:                (list or tuple) Task ids to keep track of
    :return:                        (avg_end_acc, forgetting, avg_acc_task)
    """
    n_run, n_tasks = end_task_acc_arr.shape[:2]
    t_coef = stats.t.ppf((1+0.95) / 2, n_run-1)  # t coefficient used to compute 95% CIs: mean +- t*SD, alpha/2 = 0.975

    # compute average test accuracy and CI
    end_acc = end_task_acc_arr[:, -1, :]                         # shape: (num_run, num_tasks)
    avg_acc_per_run = np.mean(end_acc, axis=1)      # mean of end task accuracies per run
    avg_end_acc = (np.mean(avg_acc_per_run), t_coef * sem(avg_acc_per_run)) if n_run > 1 else (np.mean(avg_acc_per_run),)

    # compute forgetting
    best_acc = np.max(end_task_acc_arr, axis=1)  # (num_run, num_tasks)
    final_forgets = best_acc - end_acc

    avg_fgt = np.mean(final_forgets[:, :-1], axis=1)
    avg_end_fgt = (np.mean(avg_fgt), t_coef * sem(avg_fgt)) if n_run > 1 else (np.mean(avg_fgt),)

    # compute Avg ACC on old tasks of each task
    acc_per_run = (np.sum(np.tril(end_task_acc_arr), axis=2) /
                           (np.arange(n_tasks) + 1))
    avg_acc = (np.mean(acc_per_run, axis=0), t_coef * sem(acc_per_run, axis=0)) if n_run > 1 else (np.mean(acc_per_run, axis=0),)

    # compute BWT+
    bwt_per_run = (np.sum(np.tril(end_task_acc_arr, -1), axis=(1,2)) -
                  np.sum(np.diagonal(end_task_acc_arr, axis1=1, axis2=2) *
                         (np.arange(n_tasks, 0, -1) - 1), axis=1)) / (n_tasks * (n_tasks - 1) / 2)
    bwtp_per_run = np.maximum(bwt_per_run, 0)
    avg_bwtp = (np.mean(bwtp_per_run), t_coef * sem(bwtp_per_run)) if n_run > 1 else (np.mean(bwtp_per_run),)

    # compute Avg Acc_cur (diagonal elements)
    diagonals = []
    for i in range(n_run):
        matrix = end_task_acc_arr[i]
        diagonal = np.diag(matrix)
        diagonals.append(diagonal)
    diagonal_means = [np.mean(diag) for diag in diagonals]
    avg_cur_acc = (np.mean(diagonal_means), t_coef * sem(diagonal_means)) if n_run > 1 else (np.mean(diagonal_means), )

    return avg_end_acc, avg_end_fgt, avg_cur_acc, avg_acc, avg_bwtp


def compute_performance_offline(acc_multiple_run):
    n_run = acc_multiple_run.shape[0]

    t_coef = stats.t.ppf((1 + 0.95) / 2, n_run - 1)

    # compute average test accuracy and CI
    acc = (np.mean(acc_multiple_run), t_coef * sem(acc_multiple_run)) if n_run > 1 else (np.mean(acc_multiple_run),)
    return acc


def single_run_avg_end_fgt(acc_array):
    best_acc = np.max(acc_array, axis=1)
    end_acc = acc_array[-1]
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets)
    return avg_fgt


def plot_confusion_matrix(y_true, y_pred, classes, path):
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    cf_matrix = cf_matrix / np.sum(cf_matrix, axis=1, keepdims=True)
    cf_matrix = np.around(cf_matrix, decimals=2)
    np.save(path, cf_matrix)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])

    plt.figure(figsize=(6, 6), dpi=128)
    s = sn.heatmap(df_cm, annot=True, fmt='g', cmap="coolwarm", center=0.3, square=True)
    s.set(xlabel='Prediction', ylabel='Ground truth')
    plt.savefig(path, bbox_inches='tight')

