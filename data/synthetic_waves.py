import numpy as np
import os
import pickle

# Constants
NUM_CLASSES = 4
NUM_TRAIN_SAMPLES_PER_CLASS = 500
NUM_TEST_SAMPLES_PER_CLASS = 100
SEQUENCE_LENGTH = 128
NUM_CHANNELS = 3

# Frequencies for different classes, each channel will have a pair (sin_freq, cos_freq)
frequencies = {
    0: [(1, 2), (1, 2), (1, 2)],
    1: [(1, 0), (1, 0), (0, 1)],
    2: [(0, 0), (0, 0), (0, 0)],
    3: [(1, 2), (2, 2), (2, 0)],
}

# Function to generate a single time series sample with complex pattern
def generate_sample(freq_pair, sequence_length, noise_level=0.1, normalize=False):
    t = np.linspace(0, 1, sequence_length)
    signals = []
    for sin_freq, cos_freq in freq_pair:
        sin_wave = np.sin(t * 2 * np.pi * sin_freq)
        cos_wave = np.cos(t * 2 * np.pi * cos_freq)
        signal = sin_wave + cos_wave + noise_level * np.random.randn(sequence_length)
        if normalize:
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        signals.append(signal)
    return np.column_stack(signals)

# Function to generate dataset for one class
def generate_class_samples(num_samples, class_idx, sequence_length, normalize):
    samples = []
    for _ in range(num_samples):
        sample = generate_sample(frequencies[class_idx], sequence_length, normalize=normalize)
        samples.append(sample[np.newaxis, :])
    return np.concatenate(samples)

# Function to generate the whole dataset
def generate_dataset(num_classes, num_train_samples, num_test_samples, sequence_length, normalize):
    x_train, y_train = [], []
    x_test, y_test = [], []

    for class_idx in range(num_classes):
        x_train.append(generate_class_samples(num_train_samples, class_idx, sequence_length, normalize))
        y_train.append(np.full((num_train_samples,), class_idx))

        x_test.append(generate_class_samples(num_test_samples, class_idx, sequence_length, normalize))
        y_test.append(np.full((num_test_samples,), class_idx))

    return (
        np.concatenate(x_train), np.concatenate(y_train),
        np.concatenate(x_test), np.concatenate(y_test)
    )

# Generate dataset
x_train, y_train, x_test, y_test = generate_dataset(
    NUM_CLASSES,
    NUM_TRAIN_SAMPLES_PER_CLASS,
    NUM_TEST_SAMPLES_PER_CLASS,
    SEQUENCE_LENGTH,
    normalize=True  # Set to False if you don't want to normalize the channels
)

# Now x_train, y_train, x_test, y_test are your dataset numpy arrays.
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = generate_dataset(
        NUM_CLASSES,
        NUM_TRAIN_SAMPLES_PER_CLASS,
        NUM_TEST_SAMPLES_PER_CLASS,
        SEQUENCE_LENGTH,
        normalize=False  # Set to False if you don't want to normalize the channels
    )

    ## Save signals to file
    path = './saved/sines/'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + '/x_train.pkl', 'wb') as f:
        pickle.dump(x_train, f)  # trainX tasks around 500 MB
    with open(path + '/x_test.pkl', 'wb') as f:
        pickle.dump(x_test, f)
    with open(path + '/state_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open(path + '/state_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
