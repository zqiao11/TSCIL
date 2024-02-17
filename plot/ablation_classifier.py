import numpy as np
import matplotlib.pyplot as plt

# Data
data_list = ['UCI-HAR', 'UWave', 'DSA', 'GRABMyo']
methods = ['LwF', 'MAS', 'ER']
classifiers = ['Linear (CE)', 'Linear (BCE)', 'Cosine', 'NCM']



def get_data_results(data):
    if data == 'UCI-HAR':
        bn_data = {
            'LwF': [40.00, 49.97, 45.06],
            'MAS': [47.97, 54.36, 33.94],
            'ER': [72.8, 74.91, 79.39, 71.29]
        }

        ln_data = {
            'LwF': [42.53, 54.5, 53.51],
            'MAS': [53.23, 61.43, 40.45],
            'ER': [88.9, 89.79, 90.35, 89.77]
        }

    elif data == 'UWave':
        bn_data = {
            'LwF': [47.31, 54.06, 45.55],
            'MAS': [51.94, 50.85, 25.1],
            'ER': [72.71, 68.01, 70.05, 71.87]
        }

        ln_data = {
            'LwF': [28.94, 53.04, 33.62],
            'MAS': [36.01, 34.61, 24.87],
            'ER': [80.65, 78.47, 73.73, 80.09]
        }

    elif data == 'DSA':
        bn_data = {
            'LwF': [18.49, 22.97, 20.32],
            'MAS': [31.31, 21.47, 17],
            'ER': [79.64, 82.29, 86.83, 87.97]
        }

        ln_data = {
            'LwF': [17.81, 36.53, 17.24],
            'MAS': [35.89, 27.56, 19.12],
            'ER': [97.19, 97.68, 97.4, 96.53]
        }

    elif data == 'GRABMyo':
        bn_data = {
            'LwF': [19.37, 33.87, 19.74],
            'MAS': [16.94, 19.54, 19.43],
            'ER': [46.5, 46.88, 48.77, 46.]
        }

        ln_data = {
            'LwF': [19.46, 31.11, 18.93],
            'MAS': [17.79, 21.71, 17.76],
            'ER': [60.33, 61.03, 59.58, 59.09]
        }

    return bn_data, ln_data


# Setting the positions and width for the bars
position = list(range(len(methods)))
width = 0.2
colors = ['#2878b5', '#8ECFC9', '#f8ac8c', '#c82423']

# Font sizes
tick_fontsize = 20
label_fontsize = 20
title_fontsize = 20  # Increased title font size
legend_fontsize = 18


for norm in ['BN', 'LN']:
    fig, axes = plt.subplots(1, len(data_list), figsize=(15, 5), sharey=False)
    for i, ax in enumerate(axes):
        data = data_list[i]
        bn_data, ln_data = get_data_results(data)
        results = bn_data if norm == 'BN' else ln_data
        for j, classifier in enumerate(classifiers):
            method_data = [results[method][j] if j < len(results[method]) else 0 for method in methods]
            ax.bar([p + width * j for p in position], method_data, width=width, color=colors[j], label=classifier)

        # Set y-ticks with a gap of 10
        y_ticks_max = max([max(bn_data[method]) for method in methods] + [max(ln_data[method]) for method in methods])
        y_ticks = np.arange(0, y_ticks_max + 10, 10)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{int(y)}' for y in y_ticks], fontsize=14)
        ax.set_xticks([p + 1.5 * width for p in position])
        ax.set_xticklabels(methods, fontsize=tick_fontsize)
        if i == 0:  # Only for the first subplot
            ax.set_ylabel('Final Avg Acc', fontsize=label_fontsize)
        # ax.set_xlabel('{}'.format(data), fontsize=label_fontsize)  # set font size for x-axis label
        ax.set_title(f'{data}', fontsize=title_fontsize, pad=10)  # set the title for each subplot
        ax.grid(axis='y')

    # Place a legend above the subplots
    fig.legend(classifiers, loc='upper center', ncol=len(classifiers), fontsize=legend_fontsize, frameon=False)

    # Optional: Adjust the layout
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    path = "../result/plots/ablation_cls_{}".format(norm)
    plt.savefig(path, bbox_inches='tight')
    plt.show()


# for data in data_list:
#     if data == 'UCI-HAR':
#         bn_data = {
#             'LwF': [40.00, 49.97, 45.06],
#             'MAS': [47.97, 54.36, 33.94],
#             'ER': [67.74, 75.54, 81.43, 68.9]
#         }
#
#         ln_data = {
#             'LwF': [42.53, 54.5, 53.51],
#             'MAS': [53.23, 61.43, 40.45],
#             'ER': [89.97, 89.79, 90.35, 89.77]
#         }
#
#     elif data == 'UWave':
#         bn_data = {
#             'LwF': [47.31, 54.06, 45.55],
#             'MAS': [51.94, 50.85, 25.1],
#             'ER': [69.74, 65.26, 68.33, 69.74]
#         }
#
#         ln_data = {
#             'LwF': [28.94, 53.04, 33.62],
#             'MAS': [36.01, 34.61, 24.87],
#             'ER': [78.25, 79.87, 72.39, 78.25]
#         }
#
#     elif data == 'DSA':
#         bn_data = {
#             'LwF': [18.49, 22.97, 20.32],
#             'MAS': [31.31, 21.47, 17],
#             'ER': [77.08, 76.5, 86.49, 80.24]
#         }
#
#         ln_data = {
#             'LwF': [17.81, 36.53, 17.24],
#             'MAS': [35.89, 27.56, 19.12],
#             'ER': [95.88, 97.07, 97.9, 95.39]
#         }
#
#     elif data == 'GRABMyo':
#         bn_data = {
#             'LwF': [19.37, 33.87, 19.74],
#             'MAS': [16.94, 19.54, 19.43],
#             'ER': [41.71, 45.69, 46.4, 43.09]
#         }
#
#         ln_data = {
#             'LwF': [19.46, 31.11, 18.93],
#             'MAS': [17.79, 21.71, 17.76],
#             'ER': [59.75, 59.65, 59.56, 60]
#         }
#
#     # Setting the positions and width for the bars
#     position = list(range(len(methods)))
#     width = 0.2
#     colors = ['#2878b5', '#8ECFC9', '#f8ac8c', '#c82423']
#
#     # Font sizes
#     tick_fontsize = 24
#     label_fontsize = 22
#     title_fontsize = 18
#     legend_fontsize = 24
#
#     # Plotting for each results
#     for results, title in [(bn_data, 'BN'), (ln_data, 'LN')]:
#         fig, ax = plt.subplots(figsize=(10, 7))
#         for i, classifier in enumerate(classifiers):
#             method_data = [results[method][i] if i < len(results[method]) else 0 for method in methods]
#             ax.bar([p + width * i for p in position], method_data, width=width, color=colors[i], label=classifier)
#
#         # Set y-ticks with a gap of 10
#         y_ticks_max = max([max(results[method]) for method in methods])
#         ax.set_yticks(np.arange(0, y_ticks_max + 10, 10))
#         ax.set_xticks([p + 1.5 * width for p in position])
#
#         ax.set_xticks([p + 1.5 * width for p in position])
#         ax.set_xticklabels(methods, fontsize=tick_fontsize)
#         ax.set_yticklabels([int(y) for y in ax.get_yticks().tolist()], fontsize=tick_fontsize)
#         ax.set_ylabel('Final Avg Acc', fontsize=label_fontsize)
#         ax.set_xlabel('Methods', fontsize=label_fontsize)  # set font size for x-axis label
#         # ax.set_title(f'Accuracy by method and classifier ({title})', fontsize=title_fontsize)
#         ax.grid(axis='y')
#
#         ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=legend_fontsize)
#         # if data == data_list[-1]:
#         #     ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=legend_fontsize)
#
#         plt.tight_layout()
#         path = "../result/plots/ablation_cls_{}_{}".format(data, title)
#         plt.savefig(path, bbox_inches='tight')
#         plt.show()


# Version with Confidence Interval.
#import numpy as np
# import matplotlib.pyplot as plt
#
# # Data
# data_list = ['UCI-HAR', 'UWave', 'GRABMyo']
# methods = ['LwF', 'MAS', 'ER']
# classifiers = ['Linear (CE)', 'Linear (BCE)', 'Cosine', 'NCM']
#
# for data in data_list:
#     if data == 'UCI-HAR':
#         bn_mean = {
#             'LwF': [40.00, 49.97, 45.06],
#             'MAS': [47.97, 54.36, 33.94],
#             'ER': [67.74, 75.54, 81.43, 68.9]
#         }
#
#         bn_ci = {
#             'LwF': [7.36, 11.04, 12.04],
#             'MAS': [11.4, 15.45, 2.43],
#             'ER': [23.53, 5.97, 5.28, 24.91]
#         }
#
#
#         ln_mean = {
#             'LwF': [42.53, 54.5, 53.51],
#             'MAS': [53.23, 61.43, 40.45],
#             'ER': [89.97, 89.79, 90.35, 89.77]
#         }
#
#         ln_ci = {
#             'LwF': [14.17, 12.95, 16.1],
#             'MAS': [7.48, 17.77, 13.32],
#             'ER': [1.29, 1.42, 5.28, 2.12]
#         }
#
#     elif data == 'UWave':
#         bn_mean = {
#             'LwF': [47.31, 54.06, 45.55],
#             'MAS': [51.94, 50.85, 25.1],
#             'ER': [69.74, 65.26, 68.33, 69.74]
#         }
#
#         bn_ci = {
#             'LwF': [11.11, 9.94, 7.84],
#             'MAS': [11.14, 8.99, 0.31],
#             'ER': [1.82, 2.81, 9.46, 1.82]
#         }
#
#         ln_mean = {
#             'LwF': [28.94, 53.04, 33.62],
#             'MAS': [36.01, 34.61, 24.87],
#             'ER': [78.25, 79.87, 72.39, 78.25]
#         }
#
#         ln_ci = {
#             'LwF': [4.58, 9.54, 4.82],
#             'MAS': [8.14, 9.73, 0.08],
#             'ER': [5.18, 6.04, 7.66, 5.18]
#         }
#
#     elif data == 'DSA':
#         bn_mean = {
#             'LwF': [],
#             'MAS': [],
#             'ER': []
#         }
#
#         bn_ci = {
#             'LwF': [],
#             'MAS': [],
#             'ER': []
#         }
#
#         ln_mean = {
#             'LwF': [],
#             'MAS': [],
#             'ER': []
#         }
#
#         ln_ci = {
#             'LwF': [],
#             'MAS': [],
#             'ER': []
#         }
#
#     elif data == 'GRABMyo':
#         bn_mean = {
#             'LwF': [19.14, 33.87, 19.74],
#             'MAS': [20.39, 19.54, 19.43],
#             'ER': [44.49, 45.71, 46.4, 43.09]
#         }
#
#         bn_ci = {
#             'LwF': [0.99, 4.36, 1.79],
#             'MAS': [2.8,  1.26, 3.63],
#             'ER': [2.74, 5.69, 5.74, 3.11]
#         }
#
#         ln_mean = {
#             'LwF': [19.49, 31.11, 18.93],
#             'MAS': [20.18, 21.71, 17.76],
#             'ER': [59.39, 31.35, 59.56, 59.65]
#         }
#
#         ln_ci = {
#             'LwF': [0.21, 1.7, 0.75],
#             'MAS': [2.22, 3.27, 0.66],
#             'ER': [3.05, 3.5, 2.4, 3.5]
#         }
#
#     # Setting the positions and width for the bars
#     position = list(range(len(methods)))
#     width = 0.2
#     colors = ['#2878b5', '#9ac9db', '#f8ac8c', '#c82423']
#
#     # Font sizes
#     tick_fontsize = 18
#     label_fontsize = 18
#     title_fontsize = 18
#     legend_fontsize = 14
#
#     # Plotting for each results
#     for results,  data_std, title in [(bn_mean, bn_ci, 'BN'), (ln_mean, ln_ci, 'LN')]:
#         fig, ax = plt.subplots(figsize=(10, 7))
#         for i, classifier in enumerate(classifiers):
#             method_mean = [results[method][i] if i < len(results[method]) else 0 for method in methods]
#             method_std = [data_std[method][i] if i < len(data_std[method]) else 0 for method in methods]
#             ax.bar([p + width * i for p in position], method_mean, width=width, yerr=method_std, capsize=5, color=colors[i], label=classifier)
#
#         # Set y-ticks with a gap of 10
#         y_ticks_max = max([max(results[method]) for method in methods])
#         ax.set_yticks(np.arange(0, y_ticks_max + 10, 10))
#         ax.set_xticks([p + 1.5 * width for p in position])
#
#         ax.set_xticks([p + 1.5 * width for p in position])
#         ax.set_xticklabels(methods, fontsize=tick_fontsize)
#         ax.set_yticklabels([int(y) for y in ax.get_yticks().tolist()], fontsize=tick_fontsize)
#         ax.set_ylabel('Accuracy', fontsize=label_fontsize)
#         ax.set_xlabel('Methods', fontsize=label_fontsize)  # set font size for x-axis label
#         ax.set_title(f'Accuracy by method and classifier ({title})', fontsize=title_fontsize)
#         ax.grid(axis='y')
#         ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=legend_fontsize)
#
#         plt.tight_layout()
#         # path = "../result/plots/ablation_cls_{}_{}".format(data, title)
#         # plt.savefig(path, bbox_inches='tight')
#         plt.show()

