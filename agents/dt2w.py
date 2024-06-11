import torch
import torch.nn as nn
import numpy as np
from agents.base import BaseLearner
from utils.data import Dataloader_from_numpy
from agents.utils.soft_dtw_cuda import SoftDTW
from agents.utils.functions import euclidean_dist, pod_loss_var, pod_loss_temp
from agents.lwf import loss_fn_kd


class DT2W(BaseLearner):
    """
    Class-Incremental Learning on Multivariate Time Series Via Shape-Aligned Temporal Distillation, ICASSP 2023
    """

    def __init__(self, model, args):
        super(DT2W, self).__init__(model, args)
        self.use_kd = True
        self.lambda_kd_fmap = args.lambda_kd_fmap  # fmap
        self.lambda_kd_lwf = args.lambda_kd_lwf  # prediction
        # KD on Temporal Ffeature Map:
        self.fmap_kd_metric = args.fmap_kd_metric
        print('Using {} for temporal feature map'.format(self.fmap_kd_metric))

        # ProtoAug
        self.lambda_protoAug = args.lambda_protoAug
        self.prototype = None
        self.class_label = None
        self.adaptive_weight = args.adaptive_weight

        assert args.head == 'Linear', "Currently DT2W only supports Linear single head"

    def train_epoch(self, dataloader, epoch):
        total = 0
        correct = 0
        epoch_loss_ce = 0
        epoch_loss_kd_fmap = 0
        epoch_loss_kd_pred = 0
        epoch_loss_protoAug = 0
        epoch_loss = 0
        n_old_classes = self.teacher.head.out_features if self.teacher is not None else 0
        use_cuda = True if self.device == 'cuda' else False

        if self.fmap_kd_metric == 'dtw':
            similarity_metric = SoftDTW(use_cuda=use_cuda, gamma=1, normalize=False)
        elif self.fmap_kd_metric == 'euclidean':
            similarity_metric = euclidean_dist
        elif self.fmap_kd_metric == 'pod_temporal':
            similarity_metric = pod_loss_temp
        elif self.fmap_kd_metric == 'pod_variate':
            similarity_metric = pod_loss_var
        else:
            raise ValueError("Wrong metric is given!")

        self.model.train()

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)

            if y.size == 1:
                y.unsqueeze()

            self.optimizer.zero_grad()

            # 1. Cross entropy for classification
            outputs = self.model(x)
            loss_new = self.criterion(outputs, y)

            if self.task_now == 0:
                loss_kd, loss_kd_fmap, loss_kd_pred, loss_protoAug = 0, 0, 0, 0

            else:
                # ################### Losses of KD ############################
                # 2.1 KD loss on time series feature maps
                student_fmap = self.model.feature_map(x)
                teacher_fmap = self.teacher.feature_map(x)
                if self.fmap_kd_metric == 'dtw':
                    student_fmap = student_fmap.permute(0, 2, 1)  # S-DTW requires shape of (N, L, D)
                    teacher_fmap = teacher_fmap.permute(0, 2, 1)
                loss_kd_fmap = similarity_metric(student_fmap, teacher_fmap)
                loss_kd_fmap = torch.mean(loss_kd_fmap)

                # 2.2 KD loss on predicted vectors (LwF)
                loss_kd_pred = 0
                if self.lambda_kd_lwf > 0:
                    cur_model_logits = outputs[:, :self.teacher.head.out_features]  # the logits of the old neurons
                    with torch.no_grad():
                        teacher_logits = self.teacher(x)
                    loss_kd_pred = loss_fn_kd(cur_model_logits, teacher_logits)

                loss_kd = self.lambda_kd_fmap * loss_kd_fmap + self.lambda_kd_lwf * loss_kd_pred

                # ################### 3. Loss of ProtoAug ############################
                loss_protoAug = 0
                if self.lambda_protoAug > 0:
                    proto_aug = []
                    proto_aug_label = []
                    index = list(range(n_old_classes))
                    for _ in range(self.args.batch_size):
                        np.random.shuffle(index)
                        temp = self.prototype[index[0]] + np.random.normal(0, 1, self.args.feature_dim) * self.radius
                        proto_aug.append(temp)
                        # proto_aug_label.append(self.learned_classes[index[0]])
                        proto_aug_label.append(self.class_label[index[0]])

                    proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
                    proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
                    soft_feat_aug = self.model.head(proto_aug)
                    loss_protoAug = self.criterion(soft_feat_aug, proto_aug_label)
                    loss_protoAug = self.lambda_protoAug * loss_protoAug

            if self.adaptive_weight:
                step_loss = 1 / (self.task_now + 1) * loss_new + (1 - 1 / (self.task_now + 1)) * (
                            loss_kd + loss_protoAug)
            else:
                step_loss = loss_new + loss_kd + loss_protoAug
            step_loss.backward()
            self.optimizer_step(epoch=epoch)

            epoch_loss += step_loss
            epoch_loss_ce += loss_new
            epoch_loss_kd_fmap += loss_kd_fmap
            epoch_loss_kd_pred += loss_kd_pred
            epoch_loss_protoAug += loss_protoAug
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100. * (correct / total)

        epoch_loss /= (batch_id + 1)
        epoch_loss_ce /= (batch_id + 1)
        epoch_loss_kd_fmap /= (batch_id + 1)
        epoch_loss_kd_pred /= (batch_id + 1)
        epoch_loss_protoAug /= (batch_id + 1)

        return (epoch_loss, epoch_loss_ce, epoch_loss_kd_fmap, epoch_loss_kd_pred, epoch_loss_protoAug), epoch_acc

    def epoch_loss_printer(self, epoch, acc, loss):
        print('Epoch {}/{}: Accuracy = {}, Total_loss = {}, '
              'CE = {}, DT2W = {}, LwF = {}, protoAug_loss={}'.format(epoch + 1, self.epochs,
                                                                      acc, loss[0], loss[1],
                                                                      self.lambda_kd_fmap * loss[2],
                                                                      self.lambda_kd_lwf * loss[3],
                                                                      loss[4]))

    def after_task(self, x_train, y_train):
        super(DT2W, self).after_task(x_train, y_train)
        dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
        if self.lambda_protoAug > 0:
            self.protoSave(model=self.model, loader=dataloader, current_task=self.task_now)

    def protoSave(self, model, loader, current_task):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                feature = model.feature(x.to(self.device))
                if feature.shape[0] == self.args.batch_size:
                    labels.append(y.numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)  # labels in current task
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        feature_dim = features.shape[1]

        prototype = []
        radius = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            if current_task == 0:
                cov = np.cov(feature_classwise.T)
                radius.append(np.trace(cov) / feature_dim)

        if current_task == 0:
            self.radius = np.sqrt(np.mean(radius))
            self.prototype = prototype
            self.class_label = class_label
            print(self.radius)
        else:
            self.prototype = np.concatenate((prototype, self.prototype), axis=0)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)
        model.train()

