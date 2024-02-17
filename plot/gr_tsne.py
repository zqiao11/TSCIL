import numpy as np
import torch
from utils.utils import load_pickle
from utils.data import extract_samples_according_to_labels
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.timeVAE import VariationalAutoencoderConv
from utils.setup_elements import input_size_match
from models.base import SingleHeadModel
from utils.data import Dataloader_from_numpy
from utils.utils import seed_fixer


def plot_tsne_combined(data1, labels1, data2, labels2):
    # Combine the data and labels from both groups
    combined_data = np.vstack((data1, data2))
    combined_labels = np.hstack((labels1, labels2))
    group_labels = np.hstack((np.zeros(len(data1)), np.ones(len(data2))))  # 0 for group 1, 1 for group 2

    # Perform TSNE
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30)
    combined_data_tsne = tsne.fit_transform(combined_data)

    # Define colors and markers
    colors = {0: 'tab:red', 1: 'tab:blue', 2: 'tab:green', 3: 'tab:cyan',
                     4: 'tab:pink', 5: 'tab:gray', 6: 'tab:orange', 7: 'tab:brown',
                     8: 'tab:olive', 9: 'tab:purple', 10: 'darkseagreen', 11: 'black'}
    markers = ['o', 'x']  # Markers for groups

    fig, ax = plt.subplots(figsize=(10, 6), dpi=128)
    # Plotting
    for group in [0, 1]:
        for label in np.unique(combined_labels):
            subset = combined_data_tsne[(group_labels == group) & (combined_labels == label)]
            ax.scatter(subset[:, 0], subset[:, 1], c=colors[label], marker=markers[group], s=10 if group == 0 else 15)
            # plt.scatter(subset[:, 0], subset[:, 1], c=colors[label], marker=markers[group],
            #             label=f'Group {group+1}, Class {label}')

    # ax.xlabel('TSNE Component 1')
    # ax.ylabel('TSNE Component 2')
    ax.legend()
    plt.title('TSNE visualization of raw and generated samples on UWave')
    plt.show()



seed_fixer(1234)
ckpt_vae = '/home/qzz/CNRS/MTS-CIL/result/tune_and_exp/CNN_uwave/GR_BN_Dec-18-18-54-57/vae_ckpt_r0.pt'
ckpt_model = '/home/qzz/CNRS/MTS-CIL/result/tune_and_exp/CNN_uwave/GR_BN_Dec-18-18-54-57/ckpt_r0.pt'

# ckpt_vae = '/home/qzz/CNRS/MTS-CIL/result/exp/debug/CNN_uwave/GR_BN_Dec-18-21-15-38/vae_ckpt_r0.pt'
# ckpt_model = '/home/qzz/CNRS/MTS-CIL/result/exp/debug/CNN_uwave/GR_BN_Dec-18-21-15-38/ckpt_r0.pt'
path_true = '/home/qzz/CNRS/MTS-CIL/data/saved/UWave/'

input_size = input_size_match['uwave']

vae = VariationalAutoencoderConv(seq_len=input_size[0],
                                feat_dim=input_size[1],
                                latent_dim=128,  # 2 for visualization
                                hidden_layer_sizes=[64, 128, 256, 512],  # [128, 256]
                                device='cuda',
                                recon_wt=0.1)

vae.load_state_dict(torch.load(ckpt_vae))

model = SingleHeadModel(encoder='CNN',
                        head='Linear',
                        input_channels=input_size_match['uwave'][1],
                          feature_dims=128,
                          n_layers=4,
                          seq_len=input_size_match['uwave'][0],
                          n_base_nodes=8,  # 2/8
                          norm='BN',
                          input_norm='IN',
                          dropout=0,
                          ).to('cuda')


model.load_state_dict(torch.load(ckpt_model))
model = model.eval()

order_list = [2, 1, 6, 0, 4, 5, 3, 7]
x_train = load_pickle(path_true + 'x_train.pkl')
y_train = load_pickle(path_true + 'state_train.pkl')
# x_train, y_train = extract_samples_according_to_labels(x_train, y_train, [2, 1])
y_train = np.array([order_list.index(i) for i in y_train])

dataloader = Dataloader_from_numpy(x_train, y_train, 32, shuffle=True)

features = []
features_gen = []

labels = []
labels_gen = []

for batch_id, (x, y) in enumerate(dataloader):
    x, y = x.to('cuda'), y.to('cuda')

    x_ = vae.sample(32)  # x_ is in shape of (N, C, L)
    with torch.no_grad():
        all_scores_ = model(x_.transpose(1, 2))  # model's input should be (N, L, C)
        _, y_ = torch.max(all_scores_, dim=1)

    feature = model.feature(x)
    feature_gen = model.feature(x_.transpose(1, 2))

    features.append(feature.detach().cpu().numpy())
    features_gen.append(feature_gen.detach().cpu().numpy())

    labels.append(y.detach().cpu().numpy())
    labels_gen.append(y_.detach().cpu().numpy())

features = np.concatenate(features)
features_gen = np.concatenate(features_gen)
labels = np.concatenate(labels)
labels_gen = np.concatenate(labels_gen)


plot_tsne_combined(features, labels, features_gen, labels_gen)