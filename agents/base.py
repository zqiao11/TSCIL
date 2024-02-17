# -*- coding: UTF-8 -*-
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import abc
import os
from abc import abstractmethod
from utils.data import Dataloader_from_numpy
from utils.metrics import plot_confusion_matrix
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.optimizer import set_optimizer, adjust_learning_rate
from utils.utils import EarlyStopping, BinaryCrossEntropy
from torch.optim import lr_scheduler
import copy
from agents.utils.functions import compute_cls_feature_mean_buffer


class BaseLearner(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, model: nn.Module, args: argparse.Namespace):

        super(BaseLearner, self).__init__()
        self.model = model
        self.optimizer = set_optimizer(self.model, args)
        self.scheduler = None

        self.args = args
        self.run_id = args.run_id  # index of 'run', for saving ckpt
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.device = args.device
        self.scenario = args.scenario
        self.verbose = args.verbose
        self.tsne = args.tsne
        self.cf_matrix = args.cf_matrix

        self.buffer = None
        self.er_mode = args.er_mode
        self.teacher = None
        self.use_kd = False
        self.ncm_classifier = False  # Only applicable for replay-based methods

        if not self.args.tune:
            self.ckpt_path = args.exp_path + '/ckpt_r{}.pt'.format(self.run_id)
        else:
            # To avoid conflicts between multiple running trials
            self.ckpt_path = args.exp_path + '/ckpt_{}_r{}.pt'.format(os.getpid(), self.run_id)

        self.task_now = -1  # ID of the current task

        # ToDO: Consider the case that class order can change!
        self.learned_classes = []  # Joint ohv labels for all the seen classes
        self.classes_in_task = []  # Joint ohv labels for classes in the current task

        if not self.args.early_stop:
            self.args.patience = self.epochs  # Set Early stop patience as # epochs

        if self.cf_matrix:
            self.y_pred_cf, self.y_true_cf = [], []  # Collected results for Confusion matrix

    def before_task(self, y_train):
        """
        Preparations before training a task, called in 'learn_task'.
        # Note that we assume there is no overlapping among classes across tasks.

        - update Task ID: self.task_now
        - Check the data in the new task, update the label set of learned classes.
        - Expand the model's head & update the optimizer to include the new parameters

        Args
            y_train: np array of training labels of the current task

        """
        self.task_now += 1
        self.classes_in_task = list(set(y_train.tolist()))  # labels in order, not original randomized-order labels
        n_new_classes = len(self.classes_in_task)
        assert n_new_classes > 1, "A task must contain more than 1 class"

        # ############## Single-head Model #############
        # Adapt the model and optimizer for the new task
        if self.task_now != 0:
            # self.model.increase_neurons(n_new=n_new_classes)
            self.model.update_head(n_new=n_new_classes, task_now=self.task_now)
            self.model.to(self.device)
            self.optimizer = set_optimizer(self.model, self.args, task_now=self.task_now)

        # Initialize the main criterion for classification
        if self.args.criterion == 'BCE':
            self.criterion = BinaryCrossEntropy(dim=self.model.head.out_features, device=self.device)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        if self.verbose:
            print('\n--> Task {}: {} classes in total'.format(self.task_now, len(self.learned_classes + self.classes_in_task)))

    def learn_task(self, task):
        """
        Basic workflow for learning a task. For particular methods, this function will be overwritten.
        """

        (x_train, y_train), (x_val, y_val), _ = task

        self.before_task(y_train)
        train_dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
        val_dataloader = Dataloader_from_numpy(x_val, y_val, self.batch_size, shuffle=False)
        early_stopping = EarlyStopping(path=self.ckpt_path, patience=self.args.patience, mode='min', verbose=False)
        self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                                                 steps_per_epoch=len(train_dataloader),
                                                 epochs=self.epochs,
                                                 max_lr=self.args.lr)

        for epoch in range(self.epochs):
            # Train for one epoch
            epoch_loss_train, epoch_acc_train = self.train_epoch(train_dataloader, epoch=epoch)

            # Test on val set for early stop
            epoch_loss_val, epoch_acc_val = self.cross_entropy_epoch_run(val_dataloader, mode='val')

            if self.args.lradj != 'TST':
                adjust_learning_rate(self.optimizer, self.scheduler, epoch + 1, self.args)

            if self.verbose:
                self.epoch_loss_printer(epoch, epoch_acc_train, epoch_loss_train)

            early_stopping(epoch_loss_val, self.model)
            if early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                break

        self.after_task(x_train, y_train)

    @abstractmethod
    def train_epoch(self, dataloader, epoch):
        """
        Train the agent for 1 epoch.
        Return:
            - Average Accuracy of the epoch
            - Average Loss(es) of the epoch
        """
        raise NotImplementedError

    def after_task(self, x_train, y_train):
        self.learned_classes += self.classes_in_task
        self.model.load_state_dict(torch.load(self.ckpt_path))  # eval()

        if self.buffer and self.er_mode == 'task':  # Additional pass to collect memory samples
            dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
            for batch_id, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.buffer.update(x, y)

        # Compute means of classes if using ncm classifier
        if self.ncm_classifier:
            self.means_of_exemplars = compute_cls_feature_mean_buffer(self.buffer, self.model)

        if self.use_kd:
            self.teacher = copy.deepcopy(self.model)  # eval()
            if not self.args.teacher_eval:
                self.teacher.train()


    @torch.no_grad()
    def evaluate(self, task_stream, path=None):
        """
        Evaluate on the test sets of all the learned tasks (<= task_now).
        Save the test accuracies of the learned tasks in the Acc matrix.
        Visualize the feature space with TSNE, if self.tsne == True.

        Args:
            task_stream: Object of Task Stream, list of ((x_train, y_train), (x_val, y_val), (x_test, y_test)).
            path: path prefix to save the TSNE png files.

        """
        # Get num_tasks and create Accuracy Matrix for 'val set and 'test set'
        if self.task_now == 0:
            self.num_tasks = task_stream.n_tasks
            self.Acc_tasks = {'valid': np.zeros((self.num_tasks, self.num_tasks)),
                              'test': np.zeros((self.num_tasks, self.num_tasks))}

        # Reload the original optimal model to prevent the changes of statistics in BN layers.
        self.model.load_state_dict(torch.load(self.ckpt_path))

        eval_modes = ['valid', 'test']  # 'valid' is for checking generalization.
        for mode in eval_modes:
            if self.verbose:
                print('\n ======== Evaluate on {} set ========'.format(mode))
            for i in range(self.task_now + 1):
                (x_eval, y_eval) = task_stream.tasks[i][1] if mode == 'valid' else task_stream.tasks[i][2]
                eval_dataloader_i = Dataloader_from_numpy(x_eval, y_eval, self.batch_size, shuffle=False)

                if self.cf_matrix and self.task_now+1 == self.num_tasks and mode == 'test':  # Collect results for CM
                    eval_loss_i, eval_acc_i = self.test_for_cf_matrix(eval_dataloader_i)
                else:
                    eval_loss_i, eval_acc_i = self.cross_entropy_epoch_run(eval_dataloader_i, mode='test')

                if self.verbose:
                    print('Task {}: Accuracy == {}, Test CE Loss == {} ;'.format(i, eval_acc_i, eval_loss_i))
                self.Acc_tasks[mode][self.task_now][i] = np.around(eval_acc_i, decimals=2)

                # Use test data to evaluate generator
                if self.args.agent == 'GR' and self.verbose:
                    eval_mse_loss, eval_kl_loss = self.generator.evaluate(eval_dataloader_i)
                    print('        Recons Loss (MAE) == {}, KL Div == {} ;'.format(eval_mse_loss, eval_kl_loss))

            # Print accuracy matrix of the tasks on this run
            if self.task_now + 1 == self.num_tasks and self.verbose:
                with np.printoptions(suppress=True):  # Avoid Scientific Notation
                    print('Accuracy matrix of all tasks:')
                    print(self.Acc_tasks[mode])

        # TODO: ZZ: TSNE visualization for all learned classes.
        if self.tsne and not self.args.tune:
            tsne_path = path + 't{}'.format(self.task_now)
            self.feature_space_tsne_visualization(task_stream, path=tsne_path)

        if self.args.tsne_g and self.args.agent == 'GR' and not self.args.tune:
            tsne_path = path + 't{}_g'.format(self.task_now)
            self.feature_space_tsne_visualization(task_stream, path=tsne_path, view_generator=True)

    def cross_entropy_epoch_run(self, dataloader, epoch=None, mode='train'):
        """
        Train / eval with cross entropy.

        Args:
            dataloader: dataloader for train/val/test
            epoch: used for lr_adj
            train: set True for training, False for eval

        Returns:
            epoch_loss: average cross entropy loss on this epoch
            epoch_acc: average accuracy on this epoch
        """
        total = 0
        correct = 0
        epoch_loss = 0

        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)
            if y.size == 1:
                y.unsqueeze()

            if mode == 'train':
                self.optimizer.zero_grad()
                outputs = self.model(x)
                step_loss = self.criterion(outputs, y)
                step_loss.backward()
                self.optimizer_step(epoch)

            else:
                with torch.no_grad():
                    outputs = self.model(x)
                    step_loss = self.criterion(outputs, y)

                    if self.ncm_classifier and mode == 'test':
                        features = self.model.feature(x)
                        distance = torch.cdist(F.normalize(features, p=2, dim=1),
                                               F.normalize(self.means_of_exemplars, p=2, dim=1))
                        outputs = -distance  # select the class with min distance

            epoch_loss += step_loss
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id+1)  # avg loss of a mini batch

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def test_for_cf_matrix(self, dataloader):
        """
        Test for one epoch before getting the confusion matrix.
        Run this after learning the final task.

        Args:
            dataloader: dataloader for train/test

        Returns:
            epoch_loss: average cross entropy loss on this epoch
            epoch_acc: average accuracy on this epoch
        """
        total = 0
        correct = 0
        epoch_loss = 0
        ce_loss = torch.nn.CrossEntropyLoss()
        self.model.eval()
        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)

            if y.size == 1:
                y.unsqueeze()

            with torch.no_grad():
                outputs = self.model(x)
                step_loss = ce_loss(outputs, y)

                predictions = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
                labels = y.data.cpu().numpy()
                self.y_pred_cf.extend(predictions)  # Save Prediction
                self.y_true_cf.extend(labels)  # Save Truth

            epoch_loss += step_loss
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)  # avg loss of a mini batch

        return epoch_loss, epoch_acc

    def plot_cf_matrix(self, path, classes):
        plot_confusion_matrix(self.y_true_cf, self.y_pred_cf, classes, path)

    @torch.no_grad()
    def feature_space_tsne_visualization(self, task_stream, path, view_generator=False):
        for i in range(self.task_now + 1):
            if i == 0:
                _, _, (x_all, y_all) = task_stream.tasks[i]  # Test data for visualization
            else:
                _, _, (x_i, y_i) = task_stream.tasks[i]

                x_all, y_all = np.concatenate((x_all, x_i)), np.concatenate((y_all, y_i))

        # Save the nparrays of features
        x_all = torch.Tensor(x_all).to(self.device)

        if view_generator:
            z_mean, z_log_var, z = self.generator.encoder(x_all.transpose(1, 2))
            features = z.cpu().detach().numpy()
        else:
            features = self.model.feature(x_all).cpu().detach().numpy()

        np.save(path + 'f', features)
        np.save(path + 'y', y_all)

        tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50)
        tsne_result = tsne.fit_transform(features, y_all)

        df = pd.DataFrame(tsne_result, columns=['d1', 'd2'])
        df['class'] = y_all

        plt.figure(figsize=(6, 6), dpi=128)

        color_palette = {0: 'tab:red', 1: 'tab:blue', 2: 'tab:green', 3: 'tab:cyan',
                         4: 'tab:pink', 5: 'tab:gray', 6: 'tab:orange', 7: 'tab:brown',
                         8: 'tab:olive', 9: 'tab:purple', 10: 'darkseagreen', 11: 'black'}

        g1 = sns.scatterplot(
            x="d1", y="d2",
            hue="class",
            # palette=sns.color_palette("hls", self.n_cur_classes),
            palette=color_palette,
            s=10,
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
        plt.show()

    def optimizer_step(self, epoch):

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
        self.optimizer.step()

        # if self.args.norm == 'BIN':
        #     bin_gates = [p for p in self.model.parameters() if getattr(p, 'bin_gate', False)]
        #     for p in bin_gates:
        #         p.data.clamp_(min=0, max=1)

        if self.args.lradj == 'TST':
            adjust_learning_rate(self.optimizer, self.scheduler, epoch + 1, self.args, printout=False)
            self.scheduler.step()

    def epoch_loss_printer(self, epoch, acc, loss):
        print('Epoch {}/{}: Accuracy = {}, Loss = {}'.format(epoch + 1, self.epochs, acc, loss))


class SequentialFineTune(BaseLearner):
    def __init__(self, model, args):
        super(SequentialFineTune, self).__init__(model, args)

    def train_epoch(self, dataloader, epoch):
        epoch_acc_train, epoch_loss_train, = self.cross_entropy_epoch_run(dataloader=dataloader,
                                                                          epoch=epoch,
                                                                          mode='train')
        return epoch_loss_train, epoch_acc_train

