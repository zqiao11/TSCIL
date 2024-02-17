import matplotlib.pyplot as plt
import numpy as np

# Set the number of samples, channels, and classes
num_samples = 500
num_channels = 3
num_classes = 6


def synthetic_ecg(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4 * (np.sin(x) * np.sin(0.1 * x) + 0.05 * np.random.randn(length))
    return x, y


def synthetic_eeg(length=500):
    x = np.linspace(0, 50, length)
    freq = 2 * np.pi * 0.1
    y = 0.4 * (np.sin(freq * x) + 0.5 * np.sin(2 * freq * x) + 0.1 * np.sin(4 * freq * x) + 0.05 * np.random.randn(length))
    return x, y


def synthetic_eeg_2(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4 * (np.sin(x) * np.cos(0.5 * x) + 0.05 * np.random.randn(length))
    return x, y


def synthetic_series_1(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4 * (np.sin(0.5 * x) + np.sin(0.1 * x) + 0.05 * np.random.randn(length))
    return x, y


def synthetic_series_2(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4 * (np.cos(x) + np.sin(2 * x) + 0.05 * np.random.randn(length))
    return x, y


def synthetic_series_3(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4 * (np.sin(0.1 * x) * np.cos(2 * x) + 0.05 * np.random.randn(length))
    return x, y

# Generating 12 new synthetic waveform functions based on the provided examples

def synthetic_series_4(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4 * (np.cos(2 * x) + np.sin(0.4 * x) + 0.05 * np.random.randn(length))
    return x, y

def synthetic_series_5(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4 * (np.sin(2 * np.pi * 0.05 * x) + np.cos(2 * np.pi * 0.1 * x) + 0.05 * np.random.randn(length))
    return x, y

def synthetic_series_6(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4 * (np.sin(x)**2 + np.cos(0.5 * x)**2 + 0.05 * np.random.randn(length))
    return x, y

def synthetic_series_7(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4 * (np.sin(x) * np.sin(2 * x) + np.cos(3 * x) + 0.05 * np.random.randn(length))
    return x, y

def synthetic_series_8(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4 * (np.sin(0.4 * x) * np.cos(0.1 * x) + 0.05 * np.random.randn(length))
    return x, y

def synthetic_series_9(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4 * (np.cos(2 * np.pi * 0.1 * x) + 0.5 * np.sin(2 * np.pi * 0.05 * x) + 0.05 * np.random.randn(length))
    return x, y

def synthetic_series_10(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4 * (np.sin(x) + np.sin(3 * x) + np.cos(5 * x) + 0.05 * np.random.randn(length))
    return x, y

def synthetic_series_11(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4 * (np.cos(x) * np.sin(0.3 * x) + np.sin(0.6 * x) + 0.05 * np.random.randn(length))
    return x, y

def synthetic_series_12(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4 * (np.sin(0.5 * x) * np.cos(0.45 * x) + np.cos(0.75 * x) + 0.05 * np.random.randn(length))
    return x, y

def synthetic_series_13(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4 * (np.sin(2 * np.pi * 0.4 * x) + np.cos(2 * np.pi * 0.15 * x) + 0.05 * np.random.randn(length))
    return x, y

def synthetic_series_14(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4 * (np.cos(x)**3 + np.sin(x)**3 + 0.05 * np.random.randn(length))
    return x, y

def synthetic_series_15(length=500):
    x = np.linspace(0, 50, length)
    y = 0.4*(np.sin(0.1 * x) * np.cos(0.4 * x) * np.sin(0.3 * x) + 0.05 * np.random.randn(length))
    return x, y

# These functions are designed to create a variety of waveforms with different combinations of sine and cosine functions, and random noise.

def create_distinguishable_series(length, class_id, channel_id):

    if class_id == 0:
        if channel_id == 0:
            _, series = synthetic_ecg(length)
        elif channel_id == 1:
            _, series = synthetic_eeg(length)
        elif channel_id == 2:
            _, series = synthetic_eeg_2(length)

    if class_id == 1:
        if channel_id == 0:
            _, series = synthetic_series_1(length)
        elif channel_id == 1:
            _, series = synthetic_series_2(length)
        elif channel_id == 2:
            _, series = synthetic_series_3(length)

    if class_id == 2:
        if channel_id == 0:
            _, series = synthetic_series_4(length)
        elif channel_id == 1:
            _, series = synthetic_series_5(length)
        elif channel_id == 2:
            _, series = synthetic_series_6(length)

    if class_id == 3:
        if channel_id == 0:
            _, series = synthetic_series_7(length)
        elif channel_id == 1:
            _, series = synthetic_series_8(length)
        elif channel_id == 2:
            _, series = synthetic_series_9(length)

    if class_id == 4:
        if channel_id == 0:
            _, series = synthetic_series_10(length)
        elif channel_id == 1:
            _, series = synthetic_series_11(length)
        elif channel_id == 2:
            _, series = synthetic_series_12(length)

    if class_id == 5:
        if channel_id == 0:
            _, series = synthetic_series_13(length)
        elif channel_id == 1:
            _, series = synthetic_series_14(length)
        elif channel_id == 2:
            _, series = synthetic_series_15(length)


    noise = 0.05 * np.random.randn(length)
    return series + channel_id * 2.25 + noise



# Adjust the plotting function to show the frame, change the y-axis label, and hide the title
def plot_multivariate_time_series_for_class(class_id, num_samples, num_channels, figsize, color, path):
    plt.figure(figsize=figsize)

    for channel_id in range(num_channels):
        # Generate the series for this class and channel
        series = create_distinguishable_series(num_samples, class_id, channel_id)

        # Plot the series with specified color
        plt.plot(series, color=color, linewidth=3)

    # Set the weight of the frame
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)  # Set the width of the border/frame

    # plt.xlabel('Time', fontsize=20)
    # plt.ylabel('Variable', fontsize=20)
    plt.xticks([])  # Hide x ticks
    plt.yticks([])  # Hide y ticks
    plt.grid(False)  # Hide grid
    plt.box(True)  # Show box around the plot
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.show()


for i in range(num_classes):
    # Generate and plot MTS for Class 1 with the requested changes
    path = '../result/plots/mts_cls_{}'.format(i)
    plot_multivariate_time_series_for_class(class_id=i, num_samples=num_samples, num_channels=num_channels,
                                            figsize=(6, 4.5), color='tab:blue', path=path)
