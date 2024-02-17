import torch
from agents.base import BaseLearner
from utils.data import Dataloader_from_numpy
from utils.utils import EarlyStopping
from torch.optim import lr_scheduler
from utils.optimizer import adjust_learning_rate
from utils.buffer.buffer import Buffer
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.setup_elements import *
from utils.data import extract_samples_according_to_labels
import copy
import torch.optim as optim
import random
from agents.utils.deepinversion import DeepInversionClass, get_inchannel_statistics, get_xchannel_correlations, get_inchannel_freq_statistics


class Inversion(BaseLearner):
    """
    Model Inversion to synthesize pseudo samples
    - Input initialization
    - Label space modelling
    - Optimization
    - Update buffer
    """
    def __init__(self, model, args):
        super(Inversion, self).__init__(model, args)
        args.eps_mem_batch = args.batch_size
        args.update = 'random'
        args.retrieve = 'random'
        args.buffer_tracker = False
        self.buffer = Buffer(model, args)
        assert args.er_mode == 'task', "Model Inversion cannot be online"

        parameters = dict()
        parameters["ts_channels"] = input_size_match[self.args.data][1]
        parameters["ts_length"] = input_size_match[self.args.data][0]
        parameters["save_mode"] = args.save_mode
        parameters["n_samples_to_plot"] = args.n_samples_to_plot
        parameters["iterations_per_layer"] = args.iterations_per_layer
        parameters["k_freq"] = args.k_freq  # Disable freq regularization if == 0
        parameters["regularize_freq_on_feat"] = args.regularize_freq_on_feat

        # Temporal-domain prior on input space
        parameters["inchannel_means"] = []
        parameters["inchannel_stds"] = []
        parameters["xchannel_correlations"] = []
        # Frequency-domain prior on input space
        parameters["topk_freq"] = []  # top k freq per channel for each class
        parameters["freq_means"] = [] # channel-wise means of the top k freq for each class
        parameters["freq_stds"] = []  # channel-wise stds of the top k freq for each class
        # Temporal-domain prior on feature space
        parameters["feat_inchannel_means"] = []
        parameters["feat_inchannel_stds"] = []
        parameters["feat_xchannel_correlations"] = []
        # Frequency-domain prior on feature space
        parameters["feat_topk_freq"] = []
        parameters["feat_freq_means"] = []
        parameters["feat_freq_stds"] = []
        self.parameters = parameters

        coefficients = dict()
        coefficients["lr"] = args.inversion_lr
        coefficients["main_loss_multiplier"] = 1
        coefficients["inchannel_scale"] = args.inchannel_scale
        coefficients["xchannel_scale"] = args.xchannel_scale
        coefficients["feat_scale"] = args.feat_scale  # Disable feat regularization if == 0
        self.coefficients = coefficients
        self.jitter = jitter[self.args.data]

    def learn_task(self, task):

        (x_train, y_train), (x_val, y_val), _ = task

        self.before_task(y_train)

        # Compute class-wise statistics in input space
        for i in self.classes_in_task:
            x_i, _ = extract_samples_according_to_labels(x_train, y_train, [i])
            means_i, stds_i = get_inchannel_statistics(x_i, self.device)
            correlation_matrix_i = get_xchannel_correlations(x_i, self.device)
            self.parameters["inchannel_means"].append(means_i)
            self.parameters["inchannel_stds"].append(stds_i)
            self.parameters["xchannel_correlations"].append(correlation_matrix_i)

            if self.args.k_freq != 0:
                topk_freq, freq_means_i, freq_stds_i = get_inchannel_freq_statistics(x_i, self.args.k_freq, self.device)
                self.parameters["topk_freq"].append(topk_freq)
                self.parameters["freq_means"].append(freq_means_i)
                self.parameters["freq_stds"].append(freq_stds_i)

            # # Plot real input samples
            # from agents.utils.deepinversion import save_ts_plot, create_folder
            # path = self.args.exp_path + "/gen_inputs_r{}/".format(self.run_id)
            # create_folder(path)
            # create_folder(path + "/real_inputs/")
            # for j in range(10):
            #     save_ts_plot(x_i[j], '{}/real_inputs/cls{}_id{}'.format(path, i, j))

        train_dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
        val_dataloader = Dataloader_from_numpy(x_val, y_val, self.batch_size, shuffle=False)
        early_stopping = EarlyStopping(path=self.ckpt_path, patience=self.args.patience, mode='min', verbose=False)
        self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                                                 steps_per_epoch=len(train_dataloader),
                                                 epochs=self.epochs,
                                                 max_lr=self.args.lr)

        for epoch in range(self.epochs):
            epoch_loss_train, epoch_acc_train = self.train_epoch(train_dataloader, epoch=epoch)
            epoch_loss_val, epoch_acc_val = self.cross_entropy_epoch_run(val_dataloader, mode='val')

            if self.args.lradj != 'TST':
                adjust_learning_rate(self.optimizer, self.scheduler, epoch + 1, self.args)

            if self.verbose:
                print('Epoch {}/{}: Loss = {}, Accuracy = {}'.format(epoch+1, self.epochs,
                                                                                 epoch_loss_train, epoch_acc_train))

            early_stopping(epoch_loss_val, self.model)
            if early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                break
        self.after_task(x_train, y_train)

    def train_epoch(self, dataloader, epoch):
        total = 0
        correct = 0
        epoch_loss = 0
        self.model.train()

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            if self.args.augment_batch:
                x = augment_batch(x, y, self.parameters["inchannel_stds"], self.jitter)
            total += y.size(0)

            if y.size == 1:
                y.unsqueeze()

            self.optimizer.zero_grad()
            loss_ce = 0

            if self.task_now > 0:  # Replay after 1st task
                x_buf, y_buf = self.buffer.retrieve(x=x, y=y)
                if self.args.augment_batch:
                    x_buf = augment_batch(x_buf, y_buf, self.parameters["inchannel_stds"],  self.jitter)
                outputs_buf = self.model(x_buf)
                loss_ce = self.criterion(outputs_buf, y_buf)

            outputs = self.model(x)
            loss_ce += self.criterion(outputs, y)
            loss_ce.backward()
            self.optimizer_step(epoch=epoch)

            epoch_loss += loss_ce
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)

        return epoch_loss, epoch_acc

    def after_task(self, x_train, y_train):
        self.learned_classes += self.classes_in_task
        self.model.load_state_dict(torch.load(self.ckpt_path))
        self.model.eval()
        # number of exemplars per class
        self.nb_cl_per_task = n_classes_per_task[self.args.data]
        nb_protos_cl = int(np.ceil(self.buffer.mem_size / len(self.learned_classes)))
        bs = self.nb_cl_per_task * nb_protos_cl
        criterion = nn.CrossEntropyLoss()
        path = self.args.exp_path + "/gen_inputs_r{}/".format(self.run_id)

        init_list = [] if not self.args.start_noise else None
        if not self.args.start_noise or self.args.feat_scale > 0:
            for i in self.classes_in_task:
                x_i, _ = extract_samples_according_to_labels(x_train, y_train, [i])
                x_i = torch.Tensor(x_i).to(self.device)

                # Random select samples as inputs for inversion
                if not self.args.start_noise:
                    rd_indices = torch.randint(0, x_i.shape[0], (nb_protos_cl,))
                    init_i = x_i[rd_indices]
                    init_list.append(init_i)

                # Compute prior of feature maps
                if self.args.feat_scale > 0:
                    with torch.no_grad():
                        x_i = torch.Tensor(x_i).to(self.device)
                        feature_map_i = self.model.feature_map(x_i)  # (N, D, L)
                        feature_map_i = feature_map_i.transpose(1, 2)  # (N, L, D)
                        means_i, stds_i = get_inchannel_statistics(feature_map_i, self.device)
                        correlation_matrix_i = get_xchannel_correlations(feature_map_i, self.device)
                        self.parameters["feat_inchannel_means"].append(means_i)
                        self.parameters["feat_inchannel_stds"].append(stds_i)
                        self.parameters["feat_xchannel_correlations"].append(correlation_matrix_i)

                        if self.args.k_freq != 0 and self.args.regularize_freq_on_feat:
                            feat_topk_freq, feat_freq_means_i, feat_freq_stds_i = get_inchannel_freq_statistics(
                                feature_map_i, k=self.args.k_freq, device=self.device)
                            self.parameters["feat_topk_freq"].append(feat_topk_freq)
                            self.parameters["feat_freq_means"].append(feat_freq_means_i)
                            self.parameters["feat_freq_stds"].append(feat_freq_stds_i)

        # Reorder samples in 'init' to match 'targets'
        if not self.args.start_noise:
            init = torch.zeros_like(torch.cat(init_list))
            for i in range(self.nb_cl_per_task):
                init[i::self.nb_cl_per_task, :, :] = init_list[i]
        else:
            init =None

        DeepInversionEngine = DeepInversionClass(net_teacher=self.model,
                                                 path=path,
                                                 parameters=self.parameters,
                                                 bs=bs,
                                                 jitter=self.jitter,
                                                 criterion=criterion,
                                                 coefficients=self.coefficients)

        best_inputs, targets = DeepInversionEngine.generate_batch(targets=self.classes_in_task, init=init)

        if self.args.visual_syn_feat:
            self.feature_visualization_with_synthesized_samples(x_train, y_train, best_inputs, targets,
                                                                path=path+'feat_t{}'.format(self.task_now))

        # extract old exemplars and delete part of them
        if self.task_now == 0:
            self.buffer.buffer_input = best_inputs
            self.buffer.buffer_label = targets
            self.buffer.current_index = self.buffer.buffer_input.size(0)  # Totally filled the buffer
        else:
            kept_exemplars, kept_labels = self.remove_old_exemplars(bs)
            exemplars = torch.cat((kept_exemplars, best_inputs))
            labels = torch.cat((kept_labels, targets))
            self.buffer.buffer_input = exemplars
            self.buffer.buffer_label = labels
            self.buffer.current_index = self.buffer.buffer_input.size(0)  # Totally filled the buffer

    def remove_old_exemplars(self, n_exm_per_task):
        old_exemplars = self.buffer.buffer_input
        old_labels = self.buffer.buffer_label  # Tensors now

        nb_protos_cl = int(np.ceil(get_buffer_size(self.args) / self.nb_cl_per_task / (self.task_now)))
        n_exm_per_old_task = self.nb_cl_per_task * nb_protos_cl

        kept_exemplars = []
        kept_labels = []
        for i in range(self.task_now):
            start = i * n_exm_per_old_task
            exem_i = old_exemplars[start: start + n_exm_per_task]
            labels_i = old_labels[start: start + n_exm_per_task]
            kept_exemplars.append(exem_i)
            kept_labels.append(labels_i)

        kept_exemplars = torch.cat(kept_exemplars)
        kept_labels = torch.cat(kept_labels)

        return kept_exemplars, kept_labels


    @torch.no_grad()
    def feature_visualization_with_synthesized_samples(self, x_real, y_real, x_syn, y_syn, path=None):
        from sklearn.manifold import TSNE
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        x_syn, y_syn = x_syn.data.cpu().numpy(), y_syn.data.cpu().numpy()
        x_all, y_all = np.concatenate((x_real, x_syn)), np.concatenate((y_real, y_syn))

        # Save the nparrays of features
        x_all = torch.Tensor(x_all).to(self.device)
        features = self.model.feature(x_all).cpu().detach().numpy()

        tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=50)
        tsne_result = tsne.fit_transform(features, y_all)

        df = pd.DataFrame(tsne_result, columns=['d1', 'd2'])
        df['class'] = y_all

        idx_real = np.zeros_like(y_real)
        idx_syn = np.ones_like(y_syn)
        idx_all = np.concatenate((idx_real, idx_syn))
        df['mode'] = idx_all

        plt.figure(figsize=(6, 6), dpi=128)

        g1 = sns.scatterplot(
            x="d1", y="d2",
            hue="class",
            style="mode",
            s=20,
            data=df,
            legend="full",
            alpha=1
        )

        g1.set(xticklabels=[])  # remove the tick labels
        g1.set(xlabel=None)  # remove the axis label
        g1.set(yticklabels=[])  # remove the tick labels
        g1.set(ylabel=None)  # remove the axis label
        g1.tick_params(bottom=False, left=False)  # remove the ticks
        plt.savefig(path, bbox_inches='tight')
        # plt.show()


@torch.no_grad()
def augment_batch(x, y, inchannel_stds, jitter):
    # Augmentation
    # (1) White noise
    noise = torch.randn_like(x)
    noise_strength = torch.rand(x.shape[2], device='cuda')  # random selected with an offset， 0-1
    cls_in_y = torch.unique(y)
    for i in cls_in_y:
        idx = torch.where(y == i)[0]
        noise[idx] = noise[idx] * inchannel_stds[i] * noise_strength
    x_aug = x + noise

    # (2) Random shift. Note that mean, std and correlations are not affected by shifts.
    off = random.randint(-jitter, jitter)  # apply random jitter offsets.
    x_aug = torch.roll(x_aug, shifts=off, dims=1)  # (N, L, C)

    return x_aug
