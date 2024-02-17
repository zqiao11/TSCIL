
import numpy as np
import matplotlib.pyplot as plt

# Setting a seed for reproducibility
np.random.seed(0)


def synthetic_ecg(length=500):
    x = np.linspace(0, 50, length)
    y = np.sin(x) * np.sin(0.1 * x) + 0.05 * np.random.randn(length)
    return x, y


def synthetic_eeg(length=500):
    x = np.linspace(0, 50, length)
    freq = 2 * np.pi * 0.1
    y = np.sin(freq * x) + 0.5 * np.sin(2 * freq * x) + 0.1 * np.sin(4 * freq * x) + 0.05 * np.random.randn(length)
    return x, y


def synthetic_eeg_2(length=500):
    x = np.linspace(0, 50, length)
    y = np.sin(x) * np.cos(0.5 * x) + 0.05 * np.random.randn(length)
    return x, y


def synthetic_series_1(length=500):
    x = np.linspace(0, 50, length)
    y = np.sin(0.5 * x) + np.sin(0.1 * x) + 0.05 * np.random.randn(length)
    return x, y


def synthetic_series_2(length=500):
    x = np.linspace(0, 50, length)
    y = np.cos(x) + np.sin(2 * x) + 0.05 * np.random.randn(length)
    return x, y


def synthetic_series_3(length=500):
    x = np.linspace(0, 50, length)
    y = np.sin(0.1 * x) * np.cos(2 * x) + 0.05 * np.random.randn(length)
    return x, y


def plot_pattern(pattern_label, path, length, curve_color="black", grid_color="#D3D3D3"):
    # Dictionary of functions
    functions = {
        "ecg": synthetic_ecg,
        "eeg": synthetic_eeg,
        "eeg_2": synthetic_eeg_2,
        "series_1": synthetic_series_1,
        "series_2": synthetic_series_2,
        "series_3": synthetic_series_3
    }

    if pattern_label not in functions:
        print(f"Pattern label '{pattern_label}' not recognized.")
        return

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x, y = functions[pattern_label](length)
    # ax.plot(x, y, color=curve_color, linewidth=1.5)
    ax.plot(x, y, linewidth=5)


    # Setting the grid
    ax.set_axisbelow(True)
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5, color=grid_color)
    ax.grid(True, which='both', linestyle='--', linewidth=1)
    ax.set_facecolor('#f9f9f9')
    ax.set_xlim([0, 50])
    # ax.set_xticks([])
    # ax.set_yticks([])
    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(length=0)

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.show()


length_dict = {
        "ecg": 200,
        "eeg": 200,
        "eeg_2": 200,
        "series_1": 200,
        "series_2": 200,
        "series_3": 200
    }

# Example Usage:
for i in ["ecg", "eeg", "eeg_2", "series_1", "series_2", "series_3"]:
    path = '../result/plots/cls_{}'.format(i)
    length = length_dict[i]
    plot_pattern(i, path, length, curve_color="black", grid_color="red")
