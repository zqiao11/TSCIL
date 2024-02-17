# -*- coding: UTF-8 -*-
import torch
from torch.nn import functional as F
from agents.base import BaseLearner
from utils.utils import BinaryCrossEntropy, BinaryCrossEntropywithLogits


class LwF(BaseLearner):
    def __init__(self, model, args):
        super(LwF, self).__init__(model, args)

        self.use_kd = True
        self.lwf_lambda = args.lambda_kd_lwf

    def train_epoch(self, dataloader, epoch):
        total = 0
        correct = 0
        epoch_loss_new = 0
        epoch_loss_kd = 0
        epoch_loss = 0

        self.model.train()
        num_new_cls = len(self.classes_in_task)
        num_old_cls = self.model.head.out_features - num_new_cls

        # Define loss functions for classification and KD
        if self.args.criterion == 'BCE':
            criterion_new = BinaryCrossEntropy(dim=num_new_cls, device=self.device)  # label is numeric
            criterion_kd = BinaryCrossEntropywithLogits(dim=num_old_cls, device=self.device)  # label is teacher's logits
        else:
            criterion_new = self.criterion
            criterion_kd = loss_fn_kd

        # Batch Loop
        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)

            if y.size == 1:
                y.unsqueeze()

            self.optimizer.zero_grad()
            logits = self.model(x)

            if self.args.criterion == 'BCE':
                y = y - num_old_cls  # new classes only
                logits_new = logits[:, num_old_cls:]  # sigmoid
                logtis_old = logits[:, :num_old_cls]
            else:
                logits_new = logits  # softmax
                logtis_old = logits[:, :num_old_cls]

            # Classification loss on new class
            loss_new = criterion_new(logits_new, y)

            # KD loss on old classes
            if self.teacher:
                with torch.no_grad():
                    teacher_logits = self.teacher(x)
                loss_kd = criterion_kd(logtis_old, teacher_logits)
            else:
                loss_kd = 0

            # Use the adaptive weight:
            if self.args.adaptive_weight:
                step_loss = 1 / (self.task_now+1) * loss_new + (1 - 1 / (self.task_now+1)) * self.lwf_lambda * loss_kd
            else:
                step_loss = loss_new + self.lwf_lambda * loss_kd

            step_loss.backward()
            self.optimizer_step(epoch=epoch)

            epoch_loss += step_loss
            epoch_loss_new += loss_new
            epoch_loss_kd += self.lwf_lambda * loss_kd
            prediction = torch.argmax(logits, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100. * (correct / total)

        epoch_loss /= (batch_id + 1)
        epoch_loss_new /= (batch_id + 1)
        epoch_loss_kd /= (batch_id + 1)

        return (epoch_loss, epoch_loss_new, epoch_loss_kd), epoch_acc

    def epoch_loss_printer(self, epoch, acc, loss):
        print('Epoch {}/{}: Accuracy = {}, Avg_total_loss = {}, Avg_CE_loss = {}, Avg_KD_loss = {}'.
              format(epoch + 1, self.epochs, acc, loss[0], loss[1], loss[2]))


def loss_fn_kd(scores, target_scores, T=2.):
    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # kd_loss = (-1 * targets_norm * log_scores_norm).sum(dim=1).mean() * T**2
    kd_loss = F.kl_div(log_scores_norm, targets_norm, reduction="batchmean")
    return kd_loss