
from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import torch.nn as nn
import torch.optim as optim
import collections
import random
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

"""
Adapted from https://github.com/NVlabs/DeepInversion
"""


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2])
        var = input[0].permute(1, 0, 2).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


def corrcoef(input):
    """
    Estimates the Pearson product-moment correlation coefficient matrix of the variables given by the input matrix,
    where rows are the variables and columns are the observations.
    """
    cov = torch.cov(input)
    std = torch.std(input, dim=1)
    std_mat = torch.matmul(std.unsqueeze(-1), std.unsqueeze(0))
    corrcoef_wo_nan = cov / (std_mat + 1e-10)
    return corrcoef_wo_nan

@torch.no_grad()
def get_inchannel_statistics(batch, device):
    """
    batch: time series in shape of (N, L, D), typically ts from one class
    Calculate the means and stds of D channels.
    Return: Tensors of means & stds on device, shape: (D,)
    """
    if isinstance(batch, np.ndarray):
        batch = torch.tensor(batch)

    stds, means = torch.std_mean(batch, dim=(0, 1))
    stds, means = stds.type(torch.FloatTensor).to(device),  means.type(torch.FloatTensor).to(device)

    return means, stds


@torch.no_grad()
def get_xchannel_correlations(batch, device):
    """
    batch: time series in shape of (N, L, D), typically ts from one class
    Calculate the Pearson correlation coefficient matrix of D channels.
    Return: Tensors of average coefficient matrix, shape: (D, D)
    """
    if isinstance(batch, np.ndarray):
        batch = torch.tensor(batch)
    batch = batch.transpose(1, 2)  # (N, D, L)
    correlation_matrices = torch.stack([corrcoef(x_i) for x_i in torch.unbind(batch, dim=0)], dim=0)  # (N, D, D)
    avg_correlation_matrix = torch.mean(correlation_matrices, dim=0)
    avg_correlation_matrix = avg_correlation_matrix.type(torch.FloatTensor).to(device)

    return avg_correlation_matrix


@torch.no_grad()
def get_inchannel_freq_statistics(batch, k, device):
    """
    Reference: https://github.com/AgustDD/Floss
    batch: time series in shape of (N, L, D), typically ts from one class
    Calculate the means & stds of the amplitudes of the top-k frequencies for each channel.
    If k == -1, use all the frequencies.
    Return:
        Tensor of the top-k freq per channel, (D, k)
        Tensors of channel-wise means / stds, shape: (D,)

    Note that the DC component is the same as the temporal mean. For ablation study, we separate it from here.
    """
    if isinstance(batch, np.ndarray):
        batch = torch.tensor(batch)
    batch = batch.transpose(1, 2)  # (N, D, L)
    xf = torch.fft.rfft(batch, dim=2, norm="forward")  # (N, D, floor(L/2)+1)
    xf_abs = xf.abs()  # Amplitude

    # Select the top-k frequency
    freq_list = torch.mean(xf_abs, dim=0)
    freq_list[:, 0] = 0  # Ignore the DC component
    if k == -1:
        k = xf_abs.shape[-1]
    _, topk_freq_per_channel = torch.topk(freq_list, k, dim=1)  # (D, k)

    dim_2 = torch.arange(xf_abs.shape[1]).unsqueeze(-1)
    xf_abs_topk = xf_abs[:, dim_2, topk_freq_per_channel]
    stds, means = torch.std_mean(xf_abs_topk, dim=0)
    stds, means = stds.type(torch.FloatTensor).to(device), means.type(torch.FloatTensor).to(device)

    return topk_freq_per_channel, means, stds


def inchannel_prior_loss(inputs_jit, targets, prior_means, prior_stds, alpha=1):
    """
    inputs_jit: (N, L, D)
    targets: (N, )
    alpha: coefficient to compute std of input_jit
    """
    target_classes = torch.unique(targets)
    loss_inchannel_mean, loss_inchannel_std = 0, 0

    for i in target_classes:
        inputs_i = inputs_jit[torch.where(targets == i)[0]]
        inchannel_stds, inchannel_means = torch.std_mean(inputs_i, dim=(0, 1))
        loss_inchannel_mean += torch.norm(inchannel_means - prior_means[i])
        loss_inchannel_std += torch.norm(inchannel_stds - alpha*prior_stds[i])

    loss = (loss_inchannel_mean + loss_inchannel_std) / target_classes.shape[0]
    return loss


def xchannel_prior_loss(inputs_jit, targets, prior_correlations):
    target_classes = torch.unique(targets)
    loss_xchannel_corre = 0

    for i in target_classes:
        inputs_i = inputs_jit[torch.where(targets == i)[0]]
        inputs_i = inputs_i.transpose(1, 2)
        correlation_matrices = torch.stack([corrcoef(x_i) for x_i in torch.unbind(inputs_i, dim=0)], dim=0)
        avg_correlation_matrix = torch.mean(correlation_matrices, dim=0)
        # loss_xchannel_corre += torch.norm(avg_correlation_matrix - prior_correlations[i], 'fro')
        loss_xchannel_corre += torch.norm(avg_correlation_matrix - prior_correlations[i],
                                          'fro') / avg_correlation_matrix.size(0)

    return loss_xchannel_corre / target_classes.shape[0]


def inchannel_freq_prior_loss(inputs_jit, targets, prior_means, prior_stds, topk_frq, alpha=1):
    """
    inputs_jit: (N, L, D)
    targets: (N, )
    alpha: coefficient to compute std of input_jit
    """
    target_classes = torch.unique(targets)  # Tensor, cuda?
    loss_inchannel_freq_mean, loss_inchannel_freq_std = 0, 0

    for i in target_classes:
        inputs_i = inputs_jit[torch.where(targets == i)[0]]
        inputs_i = inputs_i.transpose(1, 2)  # (N, D, L)
        xf = torch.fft.rfft(inputs_i, dim=2, norm="forward")  # (N, D, floor(L/2)+1)
        xf_abs = xf.abs()  # Amplitude
        dim_2 = torch.arange(xf_abs.shape[1]).unsqueeze(-1)
        xf_abs_topk = xf_abs[:, dim_2, topk_frq[i]]
        stds, means = torch.std_mean(xf_abs_topk, dim=0)  # (D, k)

        loss_inchannel_freq_mean += torch.norm(means - prior_means[i])
        loss_inchannel_freq_std += torch.norm(stds - prior_stds[i])

    loss = (loss_inchannel_freq_mean + loss_inchannel_freq_std) / target_classes.shape[0]
    return loss



class DeepInversionClass(object):
    def __init__(self, bs=84,
                 net_teacher=None, path="./gen_images/",
                 parameters=dict(),
                 jitter=30,
                 criterion=None,
                 coefficients=dict(),
                 network_output_function=lambda x: x,
                 hook_for_display = None):
        '''
        :param bs: batch size per GPU for image generation
        :parameter net_teacher: Pytorch model to be inverted
        :param path: path where to write temporal images and data
        :param parameters: a dictionary of control parameters
        :param jitter: amount of random shift applied to input at every iteration
        :param coefficients: dictionary with parameters and coefficients for optimization.
        network_output_function: function to be applied to the output of the network to get the output
        hook_for_display: function to be executed at every print/save call, useful to check accuracy of verifier
        '''

        print("Deep inversion class generation")
        self.net_teacher = net_teacher
        self.bs = bs
        self.save_every = 100
        self.jitter = jitter  # Related to dataset
        self.criterion = criterion
        self.network_output_function = network_output_function

        self.ts_channels = parameters["ts_channels"]
        self.ts_length = parameters["ts_length"]
        # 0: No saving; 1: save the final inputs per task;
        # 2: save the initial and final inputs; 3: save every 'save_every'
        self.save_mode = parameters["save_mode"]
        self.iterations_per_layer = parameters["iterations_per_layer"]
        self.n_samples_to_plot = parameters['n_samples_to_plot']
        self.k_freq = parameters['k_freq']

        # Temporal-domain prior on input space
        self.inchannel_means = parameters["inchannel_means"]
        self.inchannel_stds = parameters["inchannel_stds"]
        self.xchannel_correlations = parameters["xchannel_correlations"]
        # Frequency-domain prior on input space
        self.topk_freq = parameters["topk_freq"]
        self.freq_means = parameters["freq_means"]
        self.freq_stds = parameters["freq_stds"]
        # Temporal-domain prior on feature space
        self.feat_inchannel_means = parameters["feat_inchannel_means"]
        self.feat_inchannel_stds = parameters["feat_inchannel_stds"]
        self.feat_xchannel_correlations = parameters["feat_xchannel_correlations"]
        # Frequency-domain prior on feature space
        self.feat_topk_freq = parameters["feat_topk_freq"]
        self.feat_freq_means = parameters["feat_freq_means"]
        self.feat_freq_stds = parameters["feat_freq_stds"]
        self.regularize_freq_on_feat = parameters["regularize_freq_on_feat"]

        # Coefficients:
        self.lr = coefficients["lr"]
        self.main_loss_multiplier = coefficients["main_loss_multiplier"]
        self.inchannel_scale = coefficients["inchannel_scale"]
        self.xchannel_scale = coefficients["xchannel_scale"]
        self.feat_scale = coefficients["feat_scale"]
        self.num_generations = 0

        ## Create folders for images and logs
        prefix = path
        self.prefix = prefix
        create_folder(prefix)
        create_folder(prefix + "/best_inputs/")

        # # Create hooks for feature statistics
        # self.loss_r_feature_layers = []
        # for module in self.net_teacher.modules():
        #     if isinstance(module, nn.BatchNorm1d):
        #         self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))
        #
        # self.hook_for_display = None
        # if hook_for_display is not None:
        #     self.hook_for_display = hook_for_display

    def get_inputs(self, targets, init=None):
        """
        targets: list of target classes.
        """

        print("get_inputs call")

        net_teacher = self.net_teacher
        save_every = self.save_every
        best_cost = 1e4
        criterion = self.criterion

        targets = torch.LongTensor(targets * (int(self.bs / len(targets)))).to('cuda')
        data_type = torch.float

        if init == None:  # Initialized from random noise based on prior.
            # Original work is for image generation,where all values are normalized 0 to 1. We cannot do that for MTS data.
            inputs = torch.randn((self.bs, self.ts_length, self.ts_channels), device='cuda', dtype=data_type)  # (N, L, C)

            # Initialize inputs with class-wise prior (means & stds)
            # ToDo: Use inverse FFT to initialize?
            classes_in_targets = torch.unique(targets)
            sample_index_per_cls = {}
            for cls in classes_in_targets:
                idx = torch.where(targets==cls)[0]
                inputs[idx] = inputs[idx] * self.inchannel_stds[cls] + self.inchannel_means[cls]

                # Save the initialized inputs
                if self.save_mode in [2, 3]:
                    save_inputs(inputs[idx].to('cpu'),
                                path='{}/best_inputs/cls{}_0000'.format(self.prefix, cls),
                                n_samples_to_plot=self.n_samples_to_plot)
                # Save the sample indices for each class
                sample_index_per_cls[cls.item()] = idx
            inputs.requires_grad_()
        else:
            inputs = init.to(data_type)
            inputs.requires_grad_()
            classes_in_targets = torch.unique(targets)
            sample_index_per_cls = {}
            for cls in classes_in_targets:
                idx = torch.where(targets==cls)[0]
                sample_index_per_cls[cls.item()] = idx

        iteration = 0
        iterations_per_layer = self.iterations_per_layer
        optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

        for iteration_loc in range(iterations_per_layer):
            iteration += 1
            lr_scheduler(optimizer, iteration_loc, iteration_loc)

            # Augmentation
            # (1) White noise
            noise = torch.randn((self.bs, self.ts_length, self.ts_channels), device='cuda', dtype=data_type)
            noise_strength = torch.rand(self.ts_channels, device='cuda')  # random selected with an offset， 0-1
            for (cls, idx) in sample_index_per_cls.items():
                noise[idx] = noise[idx] * self.inchannel_stds[cls] * noise_strength
            inputs_jit = inputs + noise  # Std of inputs_jit is sqrt(1+noise_strength^2) * inchannel_stds

            # (2) Random shift. Note that mean, std and correlations are not affected by shifts.
            off = random.randint(-self.jitter, self.jitter)  # apply random jitter offsets.
            inputs_jit = torch.roll(inputs_jit, shifts=off, dims=1)  # (N, L, C)

            # forward pass
            optimizer.zero_grad()
            net_teacher.zero_grad()
            outputs = net_teacher(inputs_jit)
            outputs = self.network_output_function(outputs)

            # cross classification loss
            loss = criterion(outputs, targets)
            alpha = torch.sqrt(1+noise_strength**2)

            # Temporal inchannel loss
            # ToDo: Use alpha or not?
            loss_inchannel_tmp = inchannel_prior_loss(inputs_jit, targets, self.inchannel_means,
                                                      self.inchannel_stds, alpha)

            # Temporal xchannel loss
            loss_xchannel_tmp = xchannel_prior_loss(inputs_jit, targets, self.xchannel_correlations)

            # Frequency inchannel loss
            if self.k_freq != 0:
                loss_inchannel_freq = inchannel_freq_prior_loss(inputs_jit, targets, self.freq_means,
                                                                          self.freq_stds, self.topk_freq, alpha)
            else:
                loss_inchannel_freq = torch.tensor(0)

            loss_aux = self.inchannel_scale * loss_inchannel_tmp + \
                       self.inchannel_scale * loss_inchannel_freq + \
                       self.xchannel_scale * loss_xchannel_tmp

            # ToDo: 2 implementation choices:
            #  1. only compute the final feature map;
            #  2. use hook to compute all the 2D feature maps
            if self.feat_scale > 0:
                feature_maps = net_teacher.feature_map(inputs_jit).transpose(1, 2)  # (N, L, D)
                # R_feature inchannel
                loss_feat_inchannel_tmp = inchannel_prior_loss(feature_maps, targets,
                                                               self.feat_inchannel_means,
                                                               self.feat_inchannel_stds)

                # R_feature xchannel
                loss_feat_xchannel_tmp = xchannel_prior_loss(feature_maps, targets, self.feat_xchannel_correlations)

                if self.k_freq != 0 and self.regularize_freq_on_feat:
                    loss_feat_inchannel_freq = inchannel_freq_prior_loss(feature_maps, targets,
                                                                         self.feat_freq_means,
                                                                         self.feat_freq_stds,
                                                                         self.feat_topk_freq,
                                                                         alpha)
                else:
                    loss_feat_inchannel_freq = torch.tensor(0)

                loss_feat = self.inchannel_scale * loss_feat_inchannel_tmp + \
                            self.inchannel_scale * loss_feat_inchannel_freq + \
                            self.xchannel_scale * loss_feat_xchannel_tmp

                # # combining losses
                loss_aux = loss_aux + self.feat_scale * loss_feat

            loss = self.main_loss_multiplier * loss + loss_aux

            if iteration % save_every==0:
                print("------------iteration {}----------".format(iteration))
                print("total loss", loss.item())
                print("main criterion", criterion(outputs, targets).item())
                print("inchannel tmp loss", loss_inchannel_tmp.item())
                print("inchannel frq loss", loss_inchannel_freq.item())
                print("xchannel tmp loss", loss_xchannel_tmp.item())

                if self.feat_scale > 0:
                    print("feat inchannel tmp loss", loss_feat_inchannel_tmp.item())
                    print("feat inchannel frq loss", loss_feat_inchannel_freq.item())
                    print("feat xchannel tmp loss", loss_feat_xchannel_tmp.item())

                # if self.hook_for_display is not None:
                #     self.hook_for_display(inputs, targets)

            # do inputs update
            loss.backward()
            optimizer.step()

            if best_cost > loss.item() or iteration == 1:
                best_inputs = inputs.data.clone()
                best_cost = loss.item()

            if self.save_mode == 0:
                do_save = False
            elif self.save_mode in [1, 2]:
                do_save = iteration == self.iterations_per_layer
            else:
                do_save = iteration % save_every == 0

            if do_save:
                for (cls, idx) in sample_index_per_cls.items():
                    best_inputs_per_cls = best_inputs[idx].to('cpu')
                    save_inputs(best_inputs_per_cls,
                                path='{}/best_inputs/cls{}_{:04d}'.format(self.prefix, cls, iteration),
                                n_samples_to_plot=self.n_samples_to_plot)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)

        return best_inputs, targets

    def generate_batch(self, targets, init=None):
        # for ADI detach student and add put to eval mode
        net_teacher = self.net_teacher
        best_inputs, targets = self.get_inputs(targets=targets, init=init)

        net_teacher.eval()

        self.num_generations += 1

        return best_inputs, targets


def create_folder(directory):
    # from https://stackoverflow.com/a/273227
    if not os.path.exists(directory):
        os.makedirs(directory)


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def save_inputs(inputs, path, n_samples_to_plot=0):
    """
    Save the inputs of one class.
    inputs: TS from multiple classes., (N, L, C)
    path: where to store the nparray/png
    """
    inputs = np.array(inputs)
    np.save(path, inputs)

    # if n_samples_to_plot > 0, save the plots of the first n samples.
    for i in range(n_samples_to_plot):
        save_ts_plot(inputs[i], path + '_id{}'.format(i))


def save_ts_plot(ts, path, figsize=(6, 10)):
    """
    Save a plot of single ts sample.
    ts: a time series sample, (L, C)
    figsize: length x width
    """
    timesteps = np.arange(0, len(ts), 1)
    n_channels = ts.shape[1]
    fig, axes = plt.subplots(nrows=n_channels, ncols=1, figsize=figsize, dpi=128)
    for i in range(n_channels):
        axes[i].plot(timesteps, ts[:, i])
        axes[i].axis('off')
    plt.savefig(path, bbox_inches='tight')
    # plt.show()