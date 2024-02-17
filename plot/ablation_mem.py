import matplotlib.pyplot as plt
import numpy as np

# Data
data_list = ['UCI-HAR', 'UWave', 'DSA', 'GRABMyo']
agents = ['ER', 'DER', 'ASER', 'Herding', 'Offline']
memory_budget = [1, 5, 10, 20, 100]
x_positions = [1, 5, 10, 20, 50]
colors = ['#F27970', '#BB9727', '#54B345', '#8983BF']

# Font sizes
tick_fontsize = 20
label_fontsize = 20
title_fontsize = 24  # Increased title font size
legend_fontsize = 24


def get_data_results(data, norm):
    if data == 'UCI-HAR':
        # BN Data
        bn_accuracy_er = [67.4, 72.8, 73.32, 74.41, 75.39]
        bn_accuracy_der = [71.92, 74.62, 76.68, 74.85, 77.92]
        bn_accuracy_aser = [86.83, 89.06, 92.91, 91.11, 94.27]
        bn_accuracy_herding = [64.7, 71.6, 70.44, 72.23, 70.39]
        bn_accuracy_offline = 93.86

        # LN Data
        ln_accuracy_er = [85.95, 88.9, 90.77, 90.36, 92]
        ln_accuracy_der = [85.95, 90.87, 90.37, 91.92, 91.13]
        ln_accuracy_aser = [85.74, 90.3, 90.23, 90.29, 91.09]
        ln_accuracy_herding = [87.18, 89., 90.56, 91.99, 91.45]
        ln_accuracy_offline = 92.45

    elif data == 'UWave':
        # BN Data
        bn_accuracy_er = [44.42, 72.71, 84.78, 89.91, 92.17]
        bn_accuracy_der = [42.96, 69.24, 79.52, 82.71, 88.21]
        bn_accuracy_aser = [54.13, 83.18, 91.03, 93.62, 96.23]
        bn_accuracy_herding = [58.95, 78.98, 88.35, 90.72, 93.23]
        bn_accuracy_offline = 96.58
        # LN Data
        ln_accuracy_er = [48.82, 80.65, 88.42, 93.64, 94.92]
        ln_accuracy_der = [44.45, 72.44, 85.62, 92.52, 91.53]
        ln_accuracy_aser = [43.62, 77.1, 89.85, 92.17, 95.25]
        ln_accuracy_herding = [65.22, 85.21, 91.9, 93.31, 93.98]
        ln_accuracy_offline = 94.69

    elif data == 'DSA':
        # BN Data
        bn_accuracy_er = [52.36, 79.64, 82.5, 84.74, 90.93]
        bn_accuracy_der = [58.1, 62.67, 67.78, 64.62, 62.58]
        bn_accuracy_aser = [78.26, 96.89, 97.97, 98.92, 98.83]
        bn_accuracy_herding = [59.12, 80.13, 86.58, 89.18, 89.32]
        bn_accuracy_offline = 99.5

        # LN Data
        ln_accuracy_er = [85.68, 97.19, 98.89, 99.28, 99.]
        ln_accuracy_der = [84.93, 98.04, 98.92, 99.31, 98.96]
        ln_accuracy_aser = [78.44, 95.26, 98.58, 98.42, 99.28]
        ln_accuracy_herding = [92.06, 98.04, 98.74, 99.08, 99.42]
        ln_accuracy_offline = 99.78

    elif data == 'GRABMyo':
        # BN Data
        bn_accuracy_er = [32.72, 46.5, 50.62, 51.63, 53.23]
        bn_accuracy_der = [30.76, 31.4, 29.01, 30.33, 27.73]
        bn_accuracy_aser = [35.08, 55.92, 63.27, 68.23, 86.5]
        bn_accuracy_herding = [35.94, 47.92, 50.41, 51.83, 54.33]
        bn_accuracy_offline = 93.77

        # LN Data
        ln_accuracy_er = [35.9, 60.33, 69.63, 77.27, 83.52]
        ln_accuracy_der = [40.28, 64.51, 73.29, 80.3, 86.25]
        ln_accuracy_aser = [37.68, 58.14, 64.39, 69.08, 86.61]
        ln_accuracy_herding = [37.97, 60.74, 68.94, 77.02, 85.72]
        ln_accuracy_offline = 92.9

    if norm == 'BN':
        return [bn_accuracy_er, bn_accuracy_der, bn_accuracy_aser, bn_accuracy_herding, bn_accuracy_offline]
    else:
        return [ln_accuracy_er, ln_accuracy_der, ln_accuracy_aser, ln_accuracy_herding, ln_accuracy_offline]


for norm in ['BN', 'LN']:
    fig, axes = plt.subplots(1, len(data_list), figsize=(25, 5), sharey=False)
    for i, ax in enumerate(axes):
        data = data_list[i]
        results = get_data_results(data, norm)
        lines = []  # To keep track of line objects for the legend

        for j, agent in enumerate(agents):
            if agent == 'Offline':
                line, = ax.plot(x_positions[-1], results[j], marker='*', label='Offline', color='black', markersize=10)  # 16
                # Add the line object to the list for legend handling
                lines.append(line)
            else:
                line, = ax.plot(x_positions, results[j], marker='o', color=colors[j], label=agent, lw=2, markersize=6)  # 3, 10
                lines.append(line)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels=memory_budget,  fontsize=tick_fontsize)
        ax.set_ylim(20, 102)
        y_ticks = np.arange(40, 110, 20)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        if i == 0:  # Only for the first subplot
            ax.set_ylabel('Final Avg Acc', fontsize=label_fontsize)
        ax.set_xlabel("Memory Budget (%)", fontsize=label_fontsize)
        ax.set_title(f'{data}', fontsize=title_fontsize, pad=10)  # set the title for each subplot
        ax.grid(axis='y')

    # Create a custom legend
    # Extract all the other line objects except the one for 'Offline' for the legend
    legend_lines = [line for line in lines if line.get_label() != 'Offline']
    # Add a separate legend entry for 'Offline' with a single star
    legend_lines.append(
        plt.Line2D([0], [0], linestyle='none', marker='*', color='black', markersize=10, label='Offline'))
    # Create the legend with the custom entries. Place a legend above the subplots
    fig.legend(handles=legend_lines, fontsize=label_fontsize, loc='upper center', ncol=len(agents), frameon=False)

    # Optional: Adjust the layout
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    path = "../result/plots/ablation_mem_{}".format(norm)
    plt.savefig(path, bbox_inches='tight')
    plt.show()





# for data in data_list:
#     if data == 'UCI-HAR':
#         # BN Data
#         bn_accuracy_er = [66.24, 67.74, 71.13, 76.33, 72.56]
#         bn_accuracy_der = [73.14, 74.41, 72.37, 74.17, 73.78]
#         bn_accuracy_aser = [83.42, 88.03, 90.23, 89.18, 91.37]
#         bn_accuracy_herding = [68.72, 70.62, 71.05, 75.8, 72.56]
#         bn_accuracy_offline = 94.01
#
#         # LN Data
#         ln_accuracy_er = [83.39, 89.97, 90.39, 90.35, 90.22]
#         ln_accuracy_der = [87.58, 90.65, 90.9, 91.65, 90.54]
#         ln_accuracy_aser = [83.35, 89.42, 88.54, 89.17, 90.84]
#         ln_accuracy_herding = [85.51, 89.57, 91.21, 90.89, 90.22]
#         ln_accuracy_offline = 91.58
#
#     elif data == 'UWave':
#         # BN Data
#         bn_accuracy_er = [40.72, 69.74, 81.7, 87.9, 91.41]
#         bn_accuracy_der = [43.35, 71.1, 79.84, 84, 83.82]
#         bn_accuracy_aser = [52.94, 80.53, 90.55, 91.9, 94.42]
#         bn_accuracy_herding = [50.32, 75.5, 83.12, 89.05, 91.41]
#         bn_accuracy_offline = 94.72
#         # LN Data
#         ln_accuracy_er = [44.69, 78.25, 88.25, 91.85, 94.11]
#         ln_accuracy_der = [43.81, 77.17, 86.55, 88.92, 91.43]
#         ln_accuracy_aser = [39.13, 76.04, 89.59, 89.93, 94.48]
#         ln_accuracy_herding = [59.72, 86.12, 91.23, 93.99, 94.11]
#         ln_accuracy_offline = 94.69
#
#     elif data == 'DSA':
#         # BN Data
#         bn_accuracy_er = [51.24, 77.08, 82.5, 86.42, 80.29]
#         bn_accuracy_der = [54.97, 59.67, 55.56, 61.43, 60.94]
#         bn_accuracy_aser = [80.18, 94.5, 96.51, 97.74, 98.18]
#         bn_accuracy_herding = [64.44, 81.76, 86.68, 86.53, 80.29]
#         bn_accuracy_offline = 99.25
#
#         # LN Data
#         ln_accuracy_er = [83.9, 95.88, 98.24, 98.87, 98.5]
#         ln_accuracy_der = [84.87, 97.74, 98.46, 98.68, 98.71]
#         ln_accuracy_aser = [83.72, 93.88, 95.78, 97.61, 98.56]
#         ln_accuracy_herding = [90.75, 97.85, 98.35, 97.4, 98.5]
#         ln_accuracy_offline = 99.38
#
#     elif data == 'GRABMyo':
#         # BN Data
#         bn_accuracy_er = [33.74, 44.49, 47.69, 44.51, 46.75]
#         bn_accuracy_der = [27.32, 28.67, 28.11, 28.51, 26.13]  # Note to update
#         bn_accuracy_aser = [37.41, 56.82, 63.86, 65.25, 81.85]
#         bn_accuracy_herding = [34.49, 45.77, 46.17, 44.99, 46.75]
#         bn_accuracy_offline = 88.55
#
#         # LN Data
#         ln_accuracy_er = [38.87, 59.39, 66.4, 74.76, 80.72]
#         ln_accuracy_der = [41.36, 62.78, 71.21, 73.42, 77.16]  # Note to update
#         ln_accuracy_aser = [37.53, 57.9, 62.39, 64.25, 80.81]
#         ln_accuracy_herding = [40.07, 58.98, 69.59, 75.65, 80.72]
#         ln_accuracy_offline = 88.76
#
#     colors = ['#F27970', '#BB9727', '#54B345', '#8983BF']
#
#     # Plot BN Data
#     plt.figure(figsize=(8, 6))
#     plt.plot(x_positions, bn_accuracy_er, marker='o', color=colors[0], label='ER', lw=3, markersize=8)
#     plt.plot(x_positions, bn_accuracy_der, marker='o', color=colors[1], label='DER', lw=3, markersize=8)
#     plt.plot(x_positions, bn_accuracy_aser, marker='o', color=colors[2], label='ASER', lw=3, markersize=8)
#     plt.plot(x_positions, bn_accuracy_herding, marker='o', color=colors[3], label='Herding', lw=3, markersize=8)
#     plt.plot(x_positions[-1], bn_accuracy_offline, marker='*', label='Offline', color='black', markersize=10)
#     # plt.title("{}: BatchNorm".format(data), fontsize=18)
#     plt.xlabel("Memory Budget (%)", fontsize=24)
#     plt.ylabel("Final Avg Acc", fontsize=24)
#     plt.tick_params(axis='both', which='major', labelsize=16)  # Increase tick font sizes
#     plt.xticks(x_positions, memory_budget)  # Only show the ticks of 1, 5, 10, 20 and 100
#     if data == data_list[-1]:
#         plt.legend(fontsize=20)
#     plt.grid(True)
#     plt.tight_layout()
#     path = "../result/plots/ablation_mem_{}_BN".format(data)
#     plt.savefig(path, bbox_inches='tight')
#     plt.show()
#
#
#     # Plot LN Data
#     plt.figure(figsize=(8, 6))
#     plt.plot(x_positions, ln_accuracy_er, marker='o', color=colors[0], label='ER', lw=3, markersize=8)
#     plt.plot(x_positions, ln_accuracy_der, marker='o', color=colors[1], label='DER', lw=3, markersize=8)
#     plt.plot(x_positions, ln_accuracy_aser, marker='o', color=colors[2], label='ASER', lw=3, markersize=8)
#     plt.plot(x_positions, ln_accuracy_herding, marker='o', color=colors[3], label='Herding', lw=3, markersize=8)
#     plt.plot(x_positions[-1], ln_accuracy_offline, marker='*', label='Offline', color='black', markersize=10)
#     # plt.title("{}: LayerNorm".format(data), fontsize=18)
#     plt.xlabel("Memory Budget (%)", fontsize=24)
#     plt.ylabel("Final Avg Acc", fontsize=24)
#     plt.tick_params(axis='both', which='major', labelsize=16)  # Increase tick font sizes
#     plt.xticks(x_positions, memory_budget)  # Only show the ticks of 1, 5, 10, and 20
#     if data == data_list[-1]:
#         plt.legend(fontsize=20)
#     plt.grid(True)
#     plt.tight_layout()
#     path = "../result/plots/ablation_mem_{}_LN".format(data)
#     plt.savefig(path, bbox_inches='tight')
#     plt.show()
