# -*- coding: UTF-8 -*-
import torch
from agents.base import BaseLearner
from utils.data import Dataloader_from_numpy
from utils.utils import EarlyStopping
from agents.utils.functions import copy_params_dict, zerolike_params_dict
from torch.optim import lr_scheduler
from utils.optimizer import adjust_learning_rate

class MAS(BaseLearner):
    """
    https://avalanche-api.continualai.org/en/v0.2.0/_modules/avalanche/training/plugins/mas.html#MASPlugin
    """
    def __init__(self, model, args):
        super(MAS, self).__init__(model, args)

        # Regularization Parameters
        self._lambda = args.lambda_impt
        self.alpha = 0.8

        # Model parameters
        self.old_parameters = dict(copy_params_dict(self.model))
        self.importance = dict(zerolike_params_dict(self.model))

    def train_epoch(self, dataloader, epoch):
        total = 0
        correct = 0
        epoch_loss = 0

        self.model.train()
        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)

            self.optimizer.zero_grad()
            outputs = self.model(x)

            if self.task_now == 0:
                step_loss = self.criterion(outputs, y.long())
            else:
                loss_reg = 0.0

                # Apply penalty term
                for name, param in self.model.named_parameters():

                    if len(param.shape) == 0:
                        continue

                    if name in self.importance.keys():

                        if 'head' in name:
                            pass

                        loss_reg += torch.sum(
                            self.importance[name] * (param[:self.old_parameters[name].shape[0]] - self.old_parameters[name]).pow(2)
                        )
                step_loss = self.criterion(outputs, y.long()) + self._lambda * loss_reg

            step_loss.backward()
            self.optimizer_step(epoch=epoch)
            epoch_loss += step_loss
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)

        return epoch_loss, epoch_acc

    def after_task(self, x_train, y_train):
        super(MAS, self).after_task(x_train, y_train)
        dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
        self.old_parameters = dict(copy_params_dict(self.model))

        # Check if previous importance is available
        if not self.importance:
            raise ValueError("Importance is not available")

        # Get importance
        curr_importance = self._get_importance(dataloader)

        # Update importance
        for name in self.importance.keys():
            if 'head' in name and self.args.head == 'Linear':
                # For single head: MA for old weights importance, directly set the new weights
                n_old_neurons = self.importance[name].shape[0]
                old_importance = self.importance[name]
                self.importance[name] = curr_importance[name]
                self.importance[name][:n_old_neurons] = (
                        self.alpha * old_importance
                        + (1 - self.alpha) * curr_importance[name][:n_old_neurons]
                )

            # When using split heads, old cls weights are fixed. So there is no need to update their imp
            elif 'head' in name and self.args.head == 'SplitCosineLinear':
                continue

            else:
                self.importance[name] = (
                        self.alpha * self.importance[name]
                        + (1 - self.alpha) * curr_importance[name]
                )

    def _get_importance(self, dataloader):

        # Initialize importance matrix
        importance = dict(zerolike_params_dict(self.model))

        # Follow Avalanche, set model as train()
        self.model.train()

        for _, (x, _) in enumerate(dataloader):
            x = x.to(self.device)

            self.optimizer.zero_grad()

            # Since model.train(), statistics of BN layer may change
            out = self.model(x)

            # Average L2-Norm of the output
            loss = torch.norm(out, p="fro", dim=1).mean()
            loss.backward()

            # Accumulate importance
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # In multi-head architectures, the gradient is going
                    # to be None for all the heads different from the
                    # current one.
                    if param.grad is not None:
                        importance[name] += param.grad.abs()

        # Normalize importance
        importance = {
            name: importance[name] / len(dataloader)
            for name in importance.keys()
        }

        return importance
