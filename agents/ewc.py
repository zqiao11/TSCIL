# -*- coding: UTF-8 -*-
import numpy as np
import torch
import warnings
from collections import defaultdict
from agents.base import BaseLearner
from utils.data import Dataloader_from_numpy
from agents.utils.functions import copy_params_dict, zerolike_params_dict


class EWC(BaseLearner):
    """
    Modified from https://avalanche-api.continualai.org/en/v0.2.0/_modules/avalanche/training/plugins/ewc.html#EWCPlugin
    """
    def __init__(self, model, args, keep_importance_data=False):
        super(EWC, self).__init__(model, args)

        self.ewc_lambda = args.lambda_impt
        self.mode = args.ewc_mode
        self.decay_factor = 0.5
        if self.mode == "separate":
            self.keep_importance_data = True
        else:
            self.keep_importance_data = keep_importance_data
        self.saved_params = defaultdict(list)
        self.importances = defaultdict(list)

    def train_epoch(self, dataloader, epoch):
        total = 0
        correct = 0
        epoch_loss = 0
        epoch_ewc_term = 0
   
        self.model.train()

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)

            self.optimizer.zero_grad()
            outputs = self.model(x)

            if self.task_now == 0:
                step_loss = self.criterion(outputs, y.long())
            else:
                ewc_penalty = self.ewc_penalty()
                step_loss = self.criterion(outputs, y.long()) + ewc_penalty

                epoch_ewc_term += ewc_penalty

            step_loss.backward()
            self.optimizer_step(epoch=epoch)

            epoch_loss += step_loss
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)
        epoch_ewc_term /= (batch_id + 1)

        return (epoch_loss, epoch_ewc_term), epoch_acc

    def epoch_loss_printer(self, epoch, acc, loss):
        print('Epoch {}/{}: Accuracy = {}, Loss = {}, Avg_EWC_term = {}, '.format(epoch + 1, self.epochs,
                                                                                     acc, loss[0], loss[1]))

    def after_task(self, x_train, y_train):
        """
        Calculate Fisher
        :return:
        """
        super(EWC, self).after_task(x_train, y_train)
        dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
        importances = self.compute_importances(dataloader)
        self.update_importances(importances)
        self.saved_params[self.task_now] = copy_params_dict(self.model)
        # clear previous parameter values
        if self.task_now > 0 and (not self.keep_importance_data):
            del self.saved_params[self.task_now - 1]

    def ewc_penalty(self, **kwargs):
        """
        Compute EWC penalty and add it to the loss.
        """
        exp_counter = self.task_now
        if exp_counter == 0:
            return

        penalty = torch.tensor(0).float().to(self.device)

        if self.mode == "separate":
            # Loop through all the old tasks
            for experience in range(exp_counter):
                for (k1, cur_param), (k2, saved_param), (k3, imp) in zip(
                        self.model.named_parameters(),
                        self.saved_params[experience],
                        self.importances[experience],
                ):
                    assert k1 == k2 == k3, "Error: keys do not match "
                    # dynamic models may add new units
                    # new units are ignored by the regularization
                    if saved_param.size() == torch.Size():
                        pass
                    else:
                        n_units = saved_param.shape[0]
                        cur_param = cur_param[:n_units]
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()

        elif self.mode == "online":
            # Only use the penalty calculated from the last task
            prev_exp = exp_counter - 1
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    self.model.named_parameters(),
                    self.saved_params[prev_exp],
                    self.importances[prev_exp],
            ):

                n_units = saved_param.shape[0]
                cur_param = cur_param[:n_units]
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            raise ValueError("Wrong EWC mode.")

        loss_penalty = self.ewc_lambda * penalty

        return loss_penalty

    def compute_importances(self, dataloader):
        """
        Compute EWC importance matrix for each parameter
        """

        self.model.eval()
        self.criterion = torch.nn.CrossEntropyLoss()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if self.device == "cuda":
            for module in self.model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(self.model)

        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y.long())
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                    self.model.named_parameters(), importances
            ):
                assert k1 == k2
                if p.grad is not None:
                    grad = p.grad.data
                    imp += p.grad.data.clone().detach().pow(2)

        # update normalized fisher of current task
        max_fisher = max([torch.max(m) for _, m in importances])
        min_fisher = min([torch.min(m) for _, m in importances])

        # average over mini batch length
        for i in range(len(importances)):
            _, imp = importances[i]
            imp = (imp - min_fisher) / (max_fisher - min_fisher + 1e-32)
            importances[i][-1] = imp

        # # average over mini batch length
        # for _, imp in importances:
        #     imp /= float(len(dataloader))

        return importances

    @torch.no_grad()
    def update_importances(self, importances):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == "separate" or self.task_now == 0:
            self.importances[self.task_now] = importances

        elif self.mode == "online":
            for (k1, old_imp), (k2, curr_imp) in zip(
                    self.importances[self.task_now - 1], importances
            ):
                assert k1 == k2, "Error in importance computation."
                self.importances[self.task_now].append(
                    (k1, (self.decay_factor * old_imp + curr_imp))
                )

            # clear previous parameter importances
            if self.task_now > 0 and (not self.keep_importance_data):
                del self.importances[self.task_now - 1]

        else:
            raise ValueError("Wrong EWC mode.")




