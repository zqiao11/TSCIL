
from typing import Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from models.utils import *


# ############### CNN ####################
class CNNEncoder(nn.Module):
    """
    Modified from  https://github.com/emadeldeen24/AdaTime/blob/adatime_v2/models/models.py
    """
    def __init__(self, input_channels, feature_dims, norm='BN', dropout=0, hidden_dims=64, depth=3):
        super(CNNEncoder, self).__init__()

        self.feature_dims = feature_dims
        self.stacks = nn.ModuleList()
        self.in_channels = [2**(i-1) * hidden_dims if i != 0 else input_channels for i in range(depth)]

        for i in range(depth):
            if i == 0:
                stack = CNNEncoder.convStack(self.in_channels[i], hidden_dims, norm, dropout)
            elif i == depth-1:
                stack = CNNEncoder.convStack(self.in_channels[i], feature_dims, norm, dropout)
            else:
                stack = CNNEncoder.convStack(self.in_channels[i], 2 * self.in_channels[i], norm, dropout)
            self.stacks.append(stack)

    @staticmethod
    def convStack(in_channels, out_channel, norm, dropout):
        NormLayer = get_norm_layer(norm)
        stack = nn.Sequential(
            nn.Conv1d(in_channels, out_channel, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            NormLayer(out_channel),  # Input: (N, C, L)
            nn.MaxPool1d(kernel_size=2, stride=2),  # After each stack, the length will be halved.
            nn.Dropout(dropout),
        )

        return stack

    def forward(self,  x, pooling=True):
        # Input x is (N, L_in, C_in), need to transform it to (N, C_in, L_in)
        x = x.transpose(1, 2)

        for stack in self.stacks:
            x = stack(x)

        if pooling:
            return x.mean(dim=-1)  # global average pooling, (N, C_out)
        else:
            return x


# ############### TST ####################
class TSTEncoder(nn.Module):
    """
    Implementation of Naive TS Transformer, based on PatchTST codes:
    https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/layers/PatchTST_backbone.py
    Naive: point-wise input tokens + channel-mixing
    1) Not divide the sequence into patches and calculate patch embedding.
    2) Not consider each channel independently.

    input: x (bs, seq_len, in_channels)
    return:
    """

    def __init__(self, input_channels, seq_len, n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BN', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False, pe='zeros', learn_pe=True, **kwargs):
        super().__init__()

        # Input encoding
        self.seq_len = seq_len
        self.W_P = nn.Linear(input_channels, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, seq_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoderLayerStack(seq_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                            attn_dropout=attn_dropout, dropout=dropout,
                                            pre_norm=pre_norm, activation=act, res_attention=res_attention,
                                            n_layers=n_layers,
                                            store_attn=store_attn)

    def forward(self, x, pooling=True) -> Tensor:
        """

        :param x:  (bs, seq_len, in_channels)
        :return: (bs, d_model) or (bs, d_model, seq_len)
        """
        x = self.W_P(x)  # Revise W_p to fit 3d tensor                     # x: [bs x seq_len x d_model]
        x = self.dropout(x + self.W_pos)
        x = self.encoder(x)  # [bs x seq_len x d_model]

        if pooling:
            return x.mean(dim=1)  # pooling, [bs x d_model]
        else:
            return x.permute(0, 2, 1)


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BN', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoderLayerStack(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                            attn_dropout=attn_dropout, dropout=dropout,
                                            pre_norm=pre_norm, activation=act, res_attention=res_attention,
                                            n_layers=n_layers, store_attn=store_attn)

    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]

        # U is 3d, so the input to encoder is (N, L, D)

        # Encoder
        z = self.encoder(u)  # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        return z


class TSTEncoderLayerStack(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BN', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                             attn_dropout=attn_dropout, dropout=dropout,
                             activation=activation, res_attention=res_attention,
                             pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask,
                                                         attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BN', attn_dropout=0., dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                             proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)

        # if "batch" in norm.lower():
        #     self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        # else:
        #     self.norm_attn = nn.LayerNorm(d_model)

        NormLayer = get_norm_layer(norm)  # require input as (N, C, L)
        self.norm_attn = NormLayer(d_model)
        self.norm_ffn = NormLayer(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        # if "batch" in norm.lower():
        #     self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        # else:
        #     self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        src is (N, L, C). Need to transpose dims of src to fit normalization layers

        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = src.transpose(1, 2)
            src = self.norm_attn(src)
            src = src.transpose(1, 2)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,
                                                attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = src.transpose(1, 2)
            src = self.norm_attn(src)
            src = src.transpose(1, 2)

        # Feed-forward sublayer
        if self.pre_norm:
            src = src.transpose(1, 2)
            src = self.norm_ffn(src)
            src = src.transpose(1, 2)

        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = src.transpose(1, 2)
            src = self.norm_ffn(src)
            src = src.transpose(1, 2)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0.,
                 qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                   res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
                                                                         2)  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                       1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


# # https://github.com/salesforce/CoST/tree/main/models
# class TCNEncoder(nn.Module):
#     """
#     Input_linear?
#     """
#     def __init__(self, in_channels, feature_dims, norm='BN', dropout=0.2, hidden_dims=64, depth=10, extract_layers=None):
#         super().__init__()
#
#         self.feature_extractor = DilatedConvEncoder(
#             in_channels,
#             [hidden_dims] * depth + [feature_dims],
#             kernel_size=3,
#             dropout=dropout,
#             norm=norm,
#             extract_layers=extract_layers
#         )
#
#     def forward(self, x, pooling=True):
#         """
#         Input x is (N, L_in, D_in), need to transform it to (N, D_in, L_in)
#         Return a list of all fmaps after pooling layers
#         each fmap is (N, D_out, L_out)
#         """
#
#         x = x.transpose(1, 2)  # B x Ci x T
#         x = self.feature_extractor(x)  # B x Co x T
#
#         if pooling:
#             return x.mean(dim=-1)  # pooling, B x Co
#         else:
#             return x
#
#
# class DilatedConvEncoder(nn.Module):
#     def __init__(self, in_channels, channels, kernel_size, norm, dropout, extract_layers=None):
#         super().__init__()
#
#         if extract_layers is not None:
#             assert len(channels) - 1 in extract_layers
#
#         self.extract_layers = extract_layers
#         self.net = nn.Sequential(*[
#             ConvBlock(
#                 channels[i - 1] if i > 0 else in_channels,
#                 channels[i],
#                 kernel_size=kernel_size,
#                 dilation=2 ** i,
#                 dropout=dropout,
#                 norm=norm,
#                 final=(i == len(channels) - 1)
#             )
#             for i in range(len(channels))
#         ])
#
#     def forward(self, x):  # B x Ci x T
#         if self.extract_layers is not None:
#             outputs = []
#             for idx, mod in enumerate(self.net):
#                 x = mod(x)
#                 if idx in self.extract_layers:
#                     outputs.append(x)
#             return outputs
#         return self.net(x)  # B x Co x T, Co is the last value in 'channels'
#
#
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation, norm, dropout, final=False):
#         super().__init__()
#         NormLayer = get_norm_layer(norm)
#
#         self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
#         # self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.norm1 = NormLayer(out_channels)
#         self.dropout1 = nn.Dropout(dropout)
#
#         self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
#         # self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.norm2 = NormLayer(out_channels)
#         self.dropout2 = nn.Dropout(dropout)
#
#         self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
#
#     def forward(self, x):
#         residual = x if self.projector is None else self.projector(x)
#
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = F.gelu(x)
#         x = self.dropout1(x)
#
#         x = self.conv2(x)
#         x = self.norm2(x)
#         x = F.gelu(x)
#         x = self.dropout2(x)
#
#         return x + residual
#
#
# # Causal 1D CNN with Padding: https://github.com/pytorch/pytorch/issues/1333
# # Can also see the original TCN codes: # https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
# # This Conv is not causal.
# class SamePadConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
#         super().__init__()
#         self.receptive_field = (kernel_size - 1) * dilation + 1
#         padding = self.receptive_field // 2
#         self.conv = nn.Conv1d(
#             in_channels, out_channels, kernel_size,
#             padding=padding,
#             dilation=dilation,
#             groups=groups
#         )
#         self.remove = 1 if self.receptive_field % 2 == 0 else 0
#
#     def forward(self, x):
#         out = self.conv(x)
#         if self.remove > 0:
#             out = out[:, :, : -self.remove]
#         return out
