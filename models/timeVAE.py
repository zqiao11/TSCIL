import torch
import torch.nn as nn
import numpy as np
from models.utils import Flatten, Reshape
import matplotlib.pyplot as plt
from utils.utils import load_pickle, AverageMeter
from utils.data import Dataloader_from_numpy
import math
from models.utils import *


class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def forward(self, z_mean, z_log_var):
        batch, dim = z_mean.size()
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class UndoPadding1d(nn.Module):
    def __init__(self, padding=(0, 1)):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        out = x[:, :, self.padding[0]: -self.padding[-1]]
        return out


class VaeEncoder(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, hidden_layer_sizes):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.use_padding = []
        self.in_lengths = [seq_len]

        self._get_encoder()

    def _get_encoder(self):
        modules = []
        in_channels = self.feat_dim
        in_len = self.seq_len
        for i, out_channels in enumerate(self.hidden_layer_sizes):
            if in_len % 2 == 1:
                modules.append(nn.ConstantPad1d(padding=(0, 1), value=0))

            modules.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            modules.append(nn.ReLU())

            in_len = in_len // 2 if in_len % 2 == 0 else in_len // 2 + 1
            self.in_lengths.append(in_len)
            in_channels = out_channels

        self.encoder_conv = nn.Sequential(*modules)
        self.encoder_fc1 = nn.Linear(in_features=in_channels * in_len, out_features=self.latent_dim)
        self.encoder_fc2 = nn.Linear(in_features=in_channels * in_len, out_features=self.latent_dim)

    def forward(self, x):
        """
        x: (N, C, L)
        """

        hx = self.encoder_conv(x)
        hx= Flatten()(hx)
        z_mean = self.encoder_fc1(hx)
        z_log_var = self.encoder_fc2(hx)
        z = Sampling()(z_mean, z_log_var)

        return z_mean, z_log_var, z


class VaeDecoder(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, hidden_layer_sizes, in_lengths):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.in_lengths = in_lengths
        self._get_decoder()

    def _get_decoder(self):
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_layer_sizes[-1] * self.in_lengths[-1])
        modules = []
        reversed_layers = list(reversed(self.hidden_layer_sizes[:-1]))
        in_channels = self.hidden_layer_sizes[-1]
        for i, out_channels in enumerate(reversed_layers):
            modules.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
            modules.append(nn.ReLU())
            if self.in_lengths[-i-2] % 2 == 1:
                modules.append(UndoPadding1d(padding=(0, 1)))
            in_channels = out_channels
        self.decoder_conv = nn.Sequential(*modules)
        self.decoder_conv_final = nn.ConvTranspose1d(in_channels, self.feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_fc_final = nn.Linear(self.seq_len * self.feat_dim, self.seq_len * self.feat_dim)


    def forward(self, z):
        hz = self.decoder_input(z)
        hz = nn.ReLU()(hz)
        hz = Reshape(ts_channels=self.hidden_layer_sizes[-1])(hz)
        hz = self.decoder_conv(hz)
        hz = self.decoder_conv_final(hz)
        if self.seq_len % 2 == 1:
            hz = UndoPadding1d(padding=(0, 1))(hz)
        hz_flat = Flatten()(hz)
        hz_flat = self.decoder_fc_final(hz_flat)
        x_decoded = Reshape(ts_channels=self.feat_dim)(hz_flat)

        return x_decoded


class VariationalAutoencoderConv(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, hidden_layer_sizes, device, recon_wt=3.0):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.recon_wt = recon_wt

        self.total_loss_tracker = AverageMeter()
        self.recon_loss_tracker = AverageMeter()
        self.kl_loss_tracker = AverageMeter()
        self.replay_recon_loss_tracker = AverageMeter()
        self.replay_kl_loss_tracker = AverageMeter()

        self.device = device
        self.encoder = VaeEncoder(seq_len, feat_dim, latent_dim, hidden_layer_sizes).to(device)
        self.decoder = VaeDecoder(seq_len, feat_dim, latent_dim, hidden_layer_sizes, self.encoder.in_lengths).to(device)

    def forward(self, x):
        """
        x: shape of (N, C, L)
        """
        z_mean, z_log_var, z = self.encoder(x)
        x_decoded = self.decoder(z)

        return x_decoded

    def _get_recon_loss(self, x, x_recons):
        def get_reconst_loss_by_axis(x, x_c, dim):
            x_r = torch.mean(x, dim=dim)
            x_c_r = torch.mean(x_c, dim=dim)
            err = torch.square(x_r - x_c_r)
            loss = torch.sum(err)
            return loss

        # overall
        err = torch.square(x - x_recons)
        reconst_loss = torch.sum(err)  # Not mean value, but sum.
        # ToDo: Is adding this loss_by_axis a common practice for training TS VAE?
        reconst_loss += get_reconst_loss_by_axis(x, x_recons, dim=2)  # by time axis
        # reconst_loss += get_reconst_loss_by_axis(x, x_recons, axis=[1])    # by feature axis

        return reconst_loss
    
    
    def _get_loss(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        recon = self.decoder(z)
        recon_loss = self._get_recon_loss(x, recon)
        kl_loss = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))

        # kl_loss = torch.mean(torch.sum(kl_loss, dim=1)) / (self.seq_len * self.feat_dim)
        # total_loss = self.recon_wt * recon_loss + (1-self.recon_wt) * kl_loss

        kl_loss = torch.sum(torch.sum(kl_loss, dim=1))
        total_loss = self.recon_wt * recon_loss + kl_loss

        return total_loss, recon_loss, kl_loss
    
    def train_a_batch(self, x, optimizer, x_=None, rnt=0.5):
        self.train()
        optimizer.zero_grad()
        
        # Current data
        total_loss, recon_loss, kl_loss = self._get_loss(x)
        
        # Replay data
        if x_ is not None:
            total_loss_r, recon_loss_r, kl_loss_r = self._get_loss(x_)
            # total_loss = total_loss + total_loss_r
            total_loss = rnt * total_loss + (1 - rnt) * total_loss_r
        
        total_loss.backward()
        optimizer.step()

        self.total_loss_tracker.update(total_loss, x.size(0) if x_ is None else x.size(0) + x_.size(0))
        self.recon_loss_tracker.update(recon_loss, x.size(0))
        self.kl_loss_tracker.update(kl_loss, x.size(0))

        if x_ is not None:
            self.replay_recon_loss_tracker.update(recon_loss_r, x_.size(0))
            self.replay_kl_loss_tracker.update(kl_loss_r, x_.size(0))

        return {
            "loss": self.total_loss_tracker.avg(),
            "recon_loss": self.recon_loss_tracker.avg(),
            "kl_loss": self.kl_loss_tracker.avg(),
            "replay_recon_loss": self.replay_recon_loss_tracker.avg(),
            "replay_kl_loss": self.replay_kl_loss_tracker.avg(),
        }

    def sample(self, size):
        self.eval()
        z = torch.randn(size, self.latent_dim).to(self.device)
        with torch.no_grad():
            x = self.decoder(z)
        return x

    def reset_trackers(self):
        self.total_loss_tracker.reset()
        self.recon_loss_tracker.reset()
        self.kl_loss_tracker.reset()
        self.replay_recon_loss_tracker.reset()
        self.replay_kl_loss_tracker.reset()

    def _get_eval_loss(self, x):
        from agents.utils.functions import euclidean_dist

        self.eval()
        z_mean, z_log_var, z = self.encoder(x)
        recon = self.decoder(z)

        mse_loss = torch.nn.MSELoss()(x, recon)
        kl_loss = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
        kl_loss = torch.mean(torch.sum(kl_loss, dim=1)) / (self.seq_len * self.feat_dim)  # mean over batch, then divide by # of input-pixels

        return mse_loss, kl_loss

    @torch.no_grad()
    def evaluate(self, dataloader):

        """
        Compute the recons and KL div on testing data
        """
        self.eval()

        total = 0
        epoch_mse_loss = 0
        epoch_kl_loss = 0

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            x = x.transpose(1, 2)
            total += y.size(0)
            if y.size == 1:
                y.unsqueeze()

            mse_loss, kl_loss = self._get_eval_loss(x)

            epoch_mse_loss += mse_loss
            epoch_kl_loss += kl_loss

        epoch_mse_loss /= (batch_id + 1)  # avg loss of a mini batch
        epoch_kl_loss /= (batch_id + 1)

        return epoch_mse_loss, epoch_kl_loss



# def draw_orig_and_post_pred_sample(orig, reconst, n):
#
#     fig, axs = plt.subplots(n, 2, figsize=(10,6))
#     i = 1
#     for _ in range(n):
#         rnd_idx = np.random.choice(len(orig))
#         o = orig[rnd_idx]
#         r = reconst[rnd_idx]
#
#         plt.subplot(n, 2, i)
#         plt.imshow(o,
#             # cmap='gray',
#             aspect='auto')
#         # plt.title("Original")
#         i += 1
#
#         plt.subplot(n, 2, i)
#         plt.imshow(r,
#             # cmap='gray',
#             aspect='auto')
#         # plt.title("Sampled")
#         i += 1
#
#     fig.suptitle("Original vs Reconstructed Data")
#     fig.tight_layout()
#     plt.show()

# def _get_recon_loss(self, x, x_recons):
#     def get_reconst_loss_by_axis(x, x_c, dim):
#         x_r = torch.mean(x, dim=dim)
#         x_c_r = torch.mean(x_c, dim=dim)
#         err = torch.square(x_r - x_c_r)
#         loss = torch.mean(err)
#         return loss
#
#     # overall
#     err = torch.square(x - x_recons)
#     reconst_loss = torch.mean(err)
#     # ToDo: Is adding this loss_by_axis a common practice for training TS VAE?
#     reconst_loss += get_reconst_loss_by_axis(x, x_recons, dim=2)  # by time axis
#
#     return reconst_loss


#
# class VaeEncoder(nn.Module):
#     def __init__(self, seq_len, feat_dim, latent_dim, hidden_layer_sizes, input_norm):
#         super().__init__()
#         self.seq_len = seq_len
#         self.feat_dim = feat_dim
#         self.latent_dim = latent_dim
#         self.hidden_layer_sizes = hidden_layer_sizes
#         self.use_padding = []
#         self.in_lengths = [seq_len]
#
#         if input_norm == 'LN':
#             self.input_norm = nn.LayerNorm(feat_dim, elementwise_affine=False)  # Without learnable transform
#         elif input_norm == 'IN':
#             self.input_norm = TransposedInstanceNorm1d(feat_dim, affine=False)  # Not perform well on GrabMyo
#         else:
#             self.input_norm = None
#
#         self._get_encoder()
#
#
#     def _get_encoder(self):
#         modules = []
#         in_channels = self.feat_dim
#         in_len = self.seq_len
#         for i, out_channels in enumerate(self.hidden_layer_sizes):
#             if in_len % 2 == 1:
#                 modules.append(nn.ConstantPad1d(padding=(0, 1), value=0))
#
#             modules.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
#             modules.append(nn.ReLU())
#
#             in_len = in_len // 2 if in_len % 2 == 0 else in_len // 2 + 1
#             self.in_lengths.append(in_len)
#             in_channels = out_channels
#
#         self.encoder_conv = nn.Sequential(*modules)
#         self.encoder_fc1 = nn.Linear(in_features=in_channels * in_len, out_features=self.latent_dim)
#         self.encoder_fc2 = nn.Linear(in_features=in_channels * in_len, out_features=self.latent_dim)
#
#     def forward(self, x):
#         """
#         x: (N, C, L)
#         """
#
#         #  Turn unbounded x into bounded x (-1 to 1)
#         if self.input_norm:
#             x = self.input_norm(x.transpose(1, 2))
#             x = x.transpose(1, 2)
#         hx = self.encoder_conv(x)
#         hx= Flatten()(hx)
#         z_mean = self.encoder_fc1(hx)
#         z_log_var = self.encoder_fc2(hx)
#         z = Sampling()(z_mean, z_log_var)
#
#         return z_mean, z_log_var, z
#
#
# class VaeDecoder(nn.Module):
#     def __init__(self, seq_len, feat_dim, latent_dim, hidden_layer_sizes, in_lengths):
#         super().__init__()
#         self.seq_len = seq_len
#         self.feat_dim = feat_dim
#         self.latent_dim = latent_dim
#         self.hidden_layer_sizes = hidden_layer_sizes
#         self.in_lengths = in_lengths
#         self._get_decoder()
#
#     def _get_decoder(self):
#         self.decoder_input = nn.Linear(self.latent_dim, self.hidden_layer_sizes[-1] * self.in_lengths[-1])
#         modules = []
#         reversed_layers = list(reversed(self.hidden_layer_sizes[:-1]))
#         in_channels = self.hidden_layer_sizes[-1]
#         for i, out_channels in enumerate(reversed_layers):
#             modules.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
#             modules.append(nn.ReLU())
#             if self.in_lengths[-i-2] % 2 == 1:
#                 modules.append(UndoPadding1d(padding=(0, 1)))
#             in_channels = out_channels
#         self.decoder_conv = nn.Sequential(*modules)
#         self.decoder_conv_final = nn.ConvTranspose1d(in_channels, self.feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.decoder_fc_final = nn.Linear(self.seq_len * self.feat_dim, self.seq_len * self.feat_dim)
#
#
#     def forward(self, z):
#         hz = self.decoder_input(z)
#         hz = nn.ReLU()(hz)
#         hz = Reshape(ts_channels=self.hidden_layer_sizes[-1])(hz)
#         hz = self.decoder_conv(hz)
#         hz = self.decoder_conv_final(hz)
#         if self.seq_len % 2 == 1:
#             hz = UndoPadding1d(padding=(0, 1))(hz)
#         hz_flat = Flatten()(hz)
#         hz_flat = self.decoder_fc_final(hz_flat)
#         x_decoded = Reshape(ts_channels=self.feat_dim)(hz_flat)
#
#         return x_decoded
#
#
# class VariationalAutoencoderConv(nn.Module):
#     def __init__(self, seq_len, feat_dim, latent_dim, hidden_layer_sizes, input_norm, device, recon_wt=3.0):
#         super().__init__()
#         self.seq_len = seq_len
#         self.feat_dim = feat_dim
#         self.latent_dim = latent_dim
#         self.hidden_layer_sizes = hidden_layer_sizes
#         self.recon_wt = recon_wt
#
#         self.total_loss_tracker = AverageMeter()
#         self.recon_loss_tracker = AverageMeter()
#         self.kl_loss_tracker = AverageMeter()
#         self.replay_recon_loss_tracker = AverageMeter()
#         self.replay_kl_loss_tracker = AverageMeter()
#
#         self.device = device
#         self.encoder = VaeEncoder(seq_len, feat_dim, latent_dim, hidden_layer_sizes, input_norm).to(device)
#         self.decoder = VaeDecoder(seq_len, feat_dim, latent_dim, hidden_layer_sizes, self.encoder.in_lengths).to(device)
#
#     def forward(self, x):
#         """
#         x: shape of (N, C, L)
#         """
#         z_mean, z_log_var, z = self.encoder(x)
#         x_decoded = self.decoder(z)
#
#         return x_decoded
#
#     def _get_recon_loss(self, x, x_recons):
#         def get_reconst_loss_by_axis(x, x_c, axis):
#             x_r = torch.mean(x, dim=axis)
#             x_c_r = torch.mean(x_c, dim=axis)
#             err = torch.square(x_r - x_c_r)
#             loss = torch.sum(err)
#             return loss
#
#         # overall
#         err = torch.square(x - x_recons)
#         reconst_loss = torch.sum(err)  # Not mean value, but sum.
#         # ToDo: Is adding this loss_by_axis a common practice for training TS VAE?
#         reconst_loss += get_reconst_loss_by_axis(x, x_recons, axis=[2])  # by time axis
#         # reconst_loss += get_reconst_loss_by_axis(x, x_recons, axis=[1])    # by feature axis
#
#         return reconst_loss
#
#
#     def _get_loss(self, x):
#         z_mean, z_log_var, z = self.encoder(x)
#         recon = self.decoder(z)
#
#         # Recon target: normed x
#         x = self.encoder.input_norm(x.transpose(1, 2)).transpose(1, 2)
#         recon_loss = self._get_recon_loss(x, recon)
#         kl_loss = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
#         # kl_loss = torch.sum(torch.sum(kl_loss, dim=1))
#         kl_loss = torch.mean(torch.sum(kl_loss, dim=1)) / (self.seq_len * self.feat_dim)
#         total_loss = self.recon_wt * recon_loss + (1-self.recon_wt) * kl_loss
#
#         return total_loss, recon_loss, kl_loss
#
#     def train_a_batch(self, x, optimizer, x_=None, rnt=0.5):
#         self.train()
#         optimizer.zero_grad()
#
#         # Current data
#         total_loss, recon_loss, kl_loss = self._get_loss(x)
#
#         # Replay data
#         if x_ is not None:
#             total_loss_r, recon_loss_r, kl_loss_r = self._get_loss(x_)
#             # total_loss = total_loss + total_loss_r
#             total_loss = rnt * total_loss + (1 - rnt) * total_loss_r
#
#         total_loss.backward()
#         optimizer.step()
#
#         self.total_loss_tracker.update(total_loss, x.size(0) if x_ is None else x.size(0) + x_.size(0))
#         self.recon_loss_tracker.update(recon_loss, x.size(0))
#         self.kl_loss_tracker.update(kl_loss, x.size(0))
#
#         if x_ is not None:
#             self.replay_recon_loss_tracker.update(recon_loss_r, x_.size(0))
#             self.replay_kl_loss_tracker.update(kl_loss_r, x_.size(0))
#
#         return {
#             "loss": self.total_loss_tracker.avg(),
#             "recon_loss": self.recon_loss_tracker.avg(),
#             "kl_loss": self.kl_loss_tracker.avg(),
#             "replay_recon_loss": self.replay_recon_loss_tracker.avg(),
#             "replay_kl_loss": self.replay_kl_loss_tracker.avg(),
#         }
#
#     def sample(self, size):
#         self.eval()
#         z = torch.randn(size, self.latent_dim).to(self.device)
#         with torch.no_grad():
#             x = self.decoder(z)
#         return x
#
#     def reset_trackers(self):
#         self.total_loss_tracker.reset()
#         self.recon_loss_tracker.reset()
#         self.kl_loss_tracker.reset()
#         self.replay_recon_loss_tracker.reset()
#         self.replay_kl_loss_tracker.reset()
#
#     def _get_eval_loss(self, x):
#         from agents.utils.functions import euclidean_dist
#
#         self.eval()
#         z_mean, z_log_var, z = self.encoder(x)
#         recon = self.decoder(z)
#
#         x = self.encoder.input_norm(x.transpose(1, 2)).transpose(1, 2)
#         mse_loss = torch.nn.MSELoss()(x, recon)
#         kl_loss = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
#         kl_loss = torch.mean(torch.sum(kl_loss, dim=1)) / (self.seq_len * self.feat_dim)  # mean over batch, then divide by # of input-pixels
#
#         return mse_loss, kl_loss
#
#     @torch.no_grad()
#     def evaluate(self, dataloader):
#
#         """
#         Compute the recons and KL div on testing data
#         """
#         self.eval()
#
#         total = 0
#         epoch_mse_loss = 0
#         epoch_kl_loss = 0
#
#         for batch_id, (x, y) in enumerate(dataloader):
#             x, y = x.to(self.device), y.to(self.device)
#             x = x.transpose(1, 2)
#             total += y.size(0)
#             if y.size == 1:
#                 y.unsqueeze()
#
#             mse_loss, kl_loss = self._get_eval_loss(x)
#
#             epoch_mse_loss += mse_loss
#             epoch_kl_loss += kl_loss
#
#         epoch_mse_loss /= (batch_id + 1)  # avg loss of a mini batch
#         epoch_kl_loss /= (batch_id + 1)
#
#         return epoch_mse_loss, epoch_kl_loss
