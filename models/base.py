# -*- coding: UTF-8 -*-
import torch.nn as nn
from models.classifier import SingleHead, CosineLinear, SplitCosineLinear
from utils.setup_elements import input_size_match, n_classes_per_task, get_num_classes
from models.encoders import CNNEncoder, TSTEncoder
from models.utils import TransposedInstanceNorm1d


class SingleHeadModel(nn.Module):
    def __init__(self, encoder, head, input_channels, feature_dims, n_layers, seq_len, n_base_nodes, norm, input_norm, dropout):
        super(SingleHeadModel, self).__init__()


        if input_norm == 'LN':
            self.input_norm = nn.LayerNorm(input_channels, elementwise_affine=False)  # Without learnable transform
        elif input_norm == 'IN':
            self.input_norm = TransposedInstanceNorm1d(input_channels, affine=False)  # Not perform well on GrabMyo
        else:
            self.input_norm = None

        if encoder == 'CNN':
            self.encoder = CNNEncoder(input_channels, feature_dims, norm=norm, dropout=dropout, depth=n_layers)

        elif encoder == 'TST':
            self.encoder = TSTEncoder(input_channels, seq_len, n_layers=n_layers, d_model=feature_dims,
                                      d_ff=2 * feature_dims, norm=norm, attn_dropout=dropout, dropout=dropout,
                                      res_attention=True, pe='sincos')

        else:
            raise ValueError("Backbone must be CNN or TST")

        if head == 'Linear':
            self.head = SingleHead(in_features=feature_dims, out_features=n_base_nodes)
        elif head in ['CosineLinear', 'SplitCosineLinear']:
            self.head = CosineLinear(in_features=feature_dims, out_features=n_base_nodes)
        else:
            raise ValueError("Wrong head type")
        self.head_type = head

    def feature_map(self, x):
        """
        Return the feature map produced by encoder, (N, D, L)
        """
        if self.input_norm:
            x = self.input_norm(x)
        feature_map = self.encoder(x, pooling=False)
        return feature_map

    def feature(self, x):
        """
        Return the feature vector after GAP, (N, D)
        """
        if self.input_norm:
            x = self.input_norm(x)
        feature = self.encoder(x, pooling=True)
        return feature

    def forward(self, x):
        if self.input_norm:
            x = self.input_norm(x)
        x = self.encoder(x)
        x = self.head(x)
        return x

    def update_head(self, n_new, task_now=None):
        if self.head_type == 'SplitCosineLinear':
            assert task_now is not None
            assert task_now > 0
            if task_now == 1:
                in_features, out_features = self.head.in_features, self.head.out_features
                new_head = SplitCosineLinear(in_features, out_features, n_new)
                new_head.fc1.weight.data = self.head.weight.data
                new_head.sigma.data = self.head.sigma.data
                self.head = new_head
            else:
                in_features = self.head.in_features
                out_features1 = self.head.fc1.out_features
                out_features2 = self.head.fc2.out_features
                new_head = SplitCosineLinear(in_features, out_features1 + out_features2, n_new)
                new_head.fc1.weight.data[:out_features1] = self.head.fc1.weight.data
                new_head.fc1.weight.data[out_features1:] = self.head.fc2.weight.data
                new_head.sigma.data = self.head.sigma.data
                self.head = new_head
        else:
            self.head.increase_neurons(n_new)


def setup_model(args):
    Model = SingleHeadModel
    data = args.data
    n_offline_base_nodes = get_num_classes(args)

    return Model(encoder=args.encoder,
                 head=args.head,
                 input_channels=input_size_match[data][1],
                 feature_dims=args.feature_dim,
                 n_layers=args.n_layers,
                 seq_len=input_size_match[data][0],
                 n_base_nodes=n_offline_base_nodes if args.agent == 'Offline' else n_classes_per_task[data],
                 norm=args.norm,
                 input_norm=args.input_norm,
                 dropout=args.dropout,
                 ).to(args.device)

