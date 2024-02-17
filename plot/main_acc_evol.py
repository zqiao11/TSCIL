import numpy as np
import matplotlib.pyplot as plt

#F27970
#BB9727
#54B345
#32B897
#05B9E2
#8983BF
#C76DA2

#A1A9D0
#F0988C
#B883D4
#9E9E9E
#CFEAF1
#C4A5DE
#F6CAE5
#96CCCB


agents = ['Naive', 'Offline', 'LwF', 'MAS', 'DT2W', 'ER', 'DER', 'Herding', 'ASER', 'CLOPS']

colors = {'Naive': '#F27970',
          'Offline': 'black',
          'LwF': '#54B345',
          'MAS': '#32B897',
          'DT2W': '#05B9E2',
          'ER': '#BB9727',
          'DER': '#C76DA2',
          'Herding': '#A1A9D0',
          'ASER': '#F0988C',
          'CLOPS': '#96CCCB'}

linestyles = {'Naive': '-',
              'Offline': '-',
              'LwF': '-',
              'MAS': '-',
              'DT2W': '-',
              'ER': '-',
              'DER': '-',
              'Herding': '-',
              'ASER': '-',
              'CLOPS': '-'}

markers = {'Naive': 'o',
           'Offline': 'o',
           'LwF': 'o',
           'MAS': 'o',
           'DT2W': 'o',
           'ER': '^',
           'DER': '^',
           'Herding': '^',
           'ASER': '^',
           'CLOPS': '^'}

markersize = {'Naive': 8,
              'Offline': 8,
              'LwF': 8,
              'MAS': 8,
              'DT2W': 8,
              'ER': 10,
              'DER': 10,
              'Herding': 10,
              'ASER': 10,
              'CLOPS': 10}


# Font sizes
tick_fontsize = 20
label_fontsize = 20
title_fontsize = 24  # Increased title font size
legend_fontsize = 24

def get_dataset_results(data):
    if data == 'UCI-HAR':
            bn_acc = {
                'Naive': [98.74, 50.67, 32.88],
                'Offline': 93.86,
                'LwF': [98.54, 51.8,  40],
                'MAS': [98.8, 57.7, 47.97],
                'DT2W': [98.74, 64.84, 51.51],
                'ER': [98.71, 75.2,  72.8],
                'DER': [98.4,  83.43, 74.62],
                'Herding': [98.71, 72.18, 71.6],
                'ASER': [98.86, 98.94, 89.06],
                'CLOPS': [99.08, 78.38, 73.17]

            }

            ln_acc = {
                'Naive': [99.14, 54.3, 36.23],
                'Offline': 92.45,
                'LwF': [99.5,  65.13, 42.53],
                'MAS': [98.56, 75.89, 53.23],
                'DT2W': [98.59, 88.6,  77.89],
                'ER': [99.49, 97.49, 88.9],
                'DER': [99.49, 97.76, 90.87],
                'Herding': [99.49, 97.74, 89.],
                'ASER': [99.49, 98.04, 90.3],
                'CLOPS': [98.45, 96.33, 89.8]

            }

            num_tasks = 3

    elif data == 'UWave':
            bn_acc = {
                'Naive': [98.94, 49.54, 33.87, 26.03],
                'Offline': 96.58,
                'LwF': [88.89, 74.86, 58.77, 47.31],
                'MAS': [88.89, 75.78, 64.19, 51.94],
                'DT2W': [88.89, 79.77, 72.07, 62.71],
                'ER': [98.78, 89.48, 81.39, 72.71],
                'DER': [99.12, 87.4,  80.25, 69.24],
                'Herding': [98.87, 90.13, 86.03, 78.98],
                'ASER': [98.88, 95.78, 88.22, 83.18],
                'CLOPS': [98.87, 87.92, 79.51, 70.76]

            }

            ln_acc = {
                'Naive': [98.52, 49.56, 32.47, 24.84],
                'Offline': 94.69,
                'LwF': [98.52, 59.73, 38.22, 28.94],
                'MAS': [98.52, 76.36, 53.44, 36.01],
                'DT2W': [98.52, 84.92, 70.39, 58.06],
                'ER': [98.05, 94.31, 87.22, 80.65],
                'DER': [98.22, 89.25, 84. , 72.44],
                'Herding': [98.07, 95.05, 89.42, 85.21],
                'ASER': [98.11, 93.1,  85.25, 77.1],
                'CLOPS': [98.41, 90.43, 81.33, 73.96]

            }

            num_tasks = 4

    elif data == 'DSA':
            bn_acc = {
                'Naive': [1.000e+02, 5.025e+01, 3.336e+01, 2.623e+01, 2.247e+01, 1.762e+01],
                'Offline': 99.5,
                'LwF': [100., 46.67,  34.72,  26.96,  22., 18.49],
                'MAS': [100., 73.75,  47.92, 43.63, 34.12, 31.31],
                'DT2W': [100., 62.33, 43., 34.75, 31.35,  22.61],
                'ER': [100., 70.54, 63.25, 63.81, 82.73, 79.64],
                'DER': [100., 73.42, 55.97, 61.17, 58.48, 62.67],
                'Herding': [100., 79.58, 74.08, 65.23,  77.48,  80.13],
                'ASER': [100., 99.62, 98.61, 96.14, 98.3, 96.89],
                'CLOPS': [100., 83.,  65.28, 68.65, 74.72, 68.5]
            }

            ln_acc = {
                'Naive': [100., 60., 38.67, 25.81, 24., 19.92],
                'Offline': 99.8,
                'LwF': [100., 55.12, 30.03, 25.29, 21.63, 17.81],
                'MAS': [100., 79.46, 52.39, 40.25, 37.77, 35.89],
                'DT2W': [100., 59.67, 35.44, 26.35, 23.43, 21.35],
                'ER': [100., 99.83, 99.61, 98.58, 97.9, 97.19],
                'DER': [100., 99.71, 99.64, 98.5, 98.98, 98.04],
                'Herding': [100., 99.83, 99.55, 98.02, 98.33, 98.04],
                'ASER': [100., 99.04, 98.86, 98.44, 97.98, 95.26],
                'CLOPS': [100., 98.67, 96.55, 94.08, 92.92, 92.]

            }

            num_tasks = 6

    elif data == 'GRABMyo':
            bn_acc = {
                'Naive': [95.04, 47.93, 31.78, 23.87, 19.44],
                'Offline': 93.8,
                'LwF': [94.82, 48.19, 31.46, 23.69, 19.37],
                'MAS': [88.24, 48.19, 31.6,  19.98, 16.94],
                'DT2W': [95.04, 48.87, 36.21, 26.38, 20.73],
                'ER': [95.21, 63.4, 55.54, 50.99, 46.5],
                'DER': [96.75, 70.46, 55.24, 41.73, 31.4],
                'Herding': [95.33, 69.3, 59.97, 52., 47.92],
                'ASER': [96.69, 86.51, 71.16, 62.05, 55.92],
                'CLOPS': [95.78, 57.98, 54.56, 46.35, 42.74]
            }

            ln_acc = {
                'Naive': [93.2,  47.95, 31.46, 23.56, 19.39],
                'Offline': 92.9,
                'LwF': [94.06, 48.07, 31.79, 23.66, 19.46],
                'MAS': [93.2,  48.35, 31.1,  21.81, 17.79],
                'DT2W': [93.28, 57.,  37.53, 27.87, 20.75],
                'ER': [95.08, 88.59, 76.69, 67.01, 60.33],
                'DER': [95.47, 90.05, 79.41, 71.03, 64.51],
                'Herding': [94.79, 87.98, 76.73, 66.65, 60.74],
                'ASER': [94.83, 85.92, 73.64, 64.81, 58.14],
                'CLOPS': [95.3,  80.84, 65.1,  56.92, 51.08]
            }

            num_tasks = 5

    elif data == 'WISDM':
            bn_acc = {
                'Naive': [96.5,  48.44, 31.31, 23.37, 19.45, 15.54],
                'Offline': 85.74,
                'LwF': [95.64, 49.1,  32.47, 24.25, 19.3,  15.88],
                'MAS': [96.63, 49.38, 30.22, 21.76, 17.58, 11.21],
                'DT2W': [96.96, 48.24, 33.29, 24.35, 19.67, 16.37],
                'ER': [97.87, 64.1, 56.23, 49.56, 45.55, 41.68],
                'DER': [96.99, 66.1, 50.47, 39.19, 34.62, 28.13],
                'Herding': [95.31, 57.32, 53.22, 47.04, 43.52, 39.86],
                'ASER': [95.61, 80.54, 73.56, 63.23, 56.76, 51.58],
                'CLOPS': [95.06, 53.63, 45.82, 40.44, 38.11, 35.14]
            }

            ln_acc = {
                'Naive': [97.01, 49.04, 32.43, 24., 20.26, 17.42],
                'Offline': 88.7,
                'LwF': [94.17, 49.02, 36.61, 26.68, 22.4, 18.54],
                'MAS': [94.12, 53.65, 34.53, 26.58, 23.8,  18.87],
                'DT2W': [96.26, 65.6,  42.03, 33.02, 27.94, 21.44],
                'ER': [97.21, 92.61, 83.36, 77.18, 69.51, 65.42],
                'DER': [97.21, 92.58, 84.73, 76.68, 71.15, 65.76],
                'Herding': [97.37, 92.32, 86.94, 79.15, 73.7,  68.02],
                'ASER': [97.37, 83.9,  77.83, 66.23, 58.01, 52.44],
                'CLOPS': [92.98, 75.79, 64., 53.03, 47.3,  42.59]
            }

            num_tasks = 6

    return bn_acc, ln_acc, num_tasks


def plot_avg_acc_curves(data_list, norm, separate=False):

    if not separate:
        num_dataset = len(data_list)
        fig, axes = plt.subplots(nrows=1, ncols=num_dataset, figsize=(30, 5), dpi=128)
        axes = axes.flatten()
        lines = []  # To keep track of line objects for the legend

        for i in range(num_dataset):
            bn_acc, ln_acc, num_tasks = get_dataset_results(data_list[i])
            acc_collections = bn_acc if norm == 'BN' else ln_acc
            for agent, acc in acc_collections.items():
                if agent == 'Offline':
                    line, = axes[i].plot(num_tasks, acc, marker='*', label='Offline', color='black', markersize=16)
                    # Add the line object to the list for legend handling
                    if i == 0:
                        lines.append(line)
                else:
                    line, = axes[i].plot(list(range(1, 1+num_tasks)), acc, linewidth='1.8', label=agent, color=colors[agent],
                                         linestyle=linestyles[agent], marker=markers[agent], markersize=markersize[agent])
                    if i == 0:
                        lines.append(line)

                axes[i].set_title('{}'.format(data_list[i]), fontsize=title_fontsize, pad=10)
                axes[i].set_xlabel('Task Number', fontsize=label_fontsize)
                axes[i].set_xticks(list(range(1, 1+num_tasks)))
                axes[i].set_xticklabels(labels=list(range(1, 1+num_tasks)), fontsize=tick_fontsize)
                if i == 0:  # Only for the first subplot
                    axes[i].set_ylabel('Avg Acc', fontsize=label_fontsize)
                axes[i].tick_params(axis='y', labelsize=tick_fontsize)
                axes[i].set_ylim(10, 105)
                axes[i].grid(axis='y', color='gainsboro')

        # Place a legend above the subplots
        # Optional: Adjust the layout
        plt.tight_layout(rect=[0, 0, 1, 0.85])
        # fig.legend(agents, loc='upper center', ncol=len(agents), fontsize=legend_fontsize, frameon=False)

        # Create a custom legend
        # Extract all the other line objects except the one for 'Offline' for the legend
        legend_lines = [line for line in lines if line.get_label() != 'Offline']
        # Add a separate legend entry for 'Offline' with a single star
        legend_lines.append(
            plt.Line2D([0], [0], linestyle='none', marker='*', color='black', markersize=16, label='Offline'))
        # Create the legend with the custom entries. Place a legend above the subplots
        fig.legend(handles=legend_lines, fontsize=label_fontsize, loc='upper center', ncol=len(agents), frameon=False)

        plt.savefig("../result/plots/acc_evol_all_{}.png".format(norm), dpi=128, bbox_inches='tight')
        plt.show()

    else:
        num_dataset = len(data_list)
        for i in range(num_dataset):
            figure = plt.figure(figsize=(6, 6), dpi=128)
            bn_acc, ln_acc, num_tasks = get_dataset_results(data_list[i])
            acc_collections = bn_acc if norm == 'BN' else ln_acc
            for agent, acc in acc_collections.items():
                if agent == 'Offline':
                    plt.plot(num_tasks, acc, marker='*', label='Offline', color='black', markersize=10)
                else:
                    plt.plot(list(range(1, 1+num_tasks)), acc, linewidth='1.8', label=agent, color=colors[agent],
                                 linestyle=linestyles[agent], marker=markers[agent], markersize=markersize[agent])

            # plt.title("{}: BatchNorm".format(data), fontsize=18)
            plt.xlabel('Task Number', fontsize=24)
            plt.ylabel('Avg Acc', fontsize=22)
            plt.tick_params(axis='both', which='major', labelsize=12)  # Increase tick font sizes
            plt.grid(color='gainsboro')
            plt.xticks(list(range(1, 1+num_tasks)))
            if i == 0:
                plt.legend(fontsize=14)
            plt.savefig("../result/plots/acc_evol_{}_{}.png".format(data_list[i], norm), dpi=128, bbox_inches='tight')
            plt.show()


if __name__ == '__main__':
    # ['UCI-HAR', 'UWave', 'DSA', 'GRABMyo', 'WISDM']
    plot_avg_acc_curves(['UCI-HAR', 'UWave', 'DSA', 'GRABMyo', 'WISDM'], norm='BN', separate=False)
    plot_avg_acc_curves(['UCI-HAR', 'UWave', 'DSA', 'GRABMyo', 'WISDM'], norm='LN', separate=False)
