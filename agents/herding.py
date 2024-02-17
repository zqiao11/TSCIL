import torch
import numpy as np
from agents.base import BaseLearner
from utils.data import Dataloader_from_numpy
from utils.buffer.buffer import Buffer
import torch.nn as nn
import torch.nn.functional as F
from utils.setup_elements import *
from utils.data import extract_samples_according_to_labels
import copy
import torch.optim as optim
from agents.utils.functions import compute_features, compute_cls_feature_mean_buffer


class Herding(BaseLearner):
    """
    Class for Herding / iCarL / LUCIR / Herding

    Based on https://github.com/yaoyao-liu/class-incremental-learning
    """

    def __init__(self, model, args):
        super(Herding, self).__init__(model, args)

        args.eps_mem_batch = args.batch_size
        # If using custom retrieve/update strategies, set args.retrieve and args.update to 'random'.
        args.retrieve = 'random'
        args.update = 'random'
        self.buffer = Buffer(model, args)
        self.ncm_classifier = args.ncm_classifier
        print('ER mode: {}, NCM classifier: {}'.format(self.er_mode, self.ncm_classifier))

        self.mnemonics = True if args.agent == 'Mnemonics' else False

        print('Using Herding for sample selection')
        self.dictionary_size = n_smp_per_cls[args.data]  # To compute class feature mean, must be larger than X.shape[0]
        self.nb_cl_per_task = n_classes_per_task[args.data]
        num_classes = get_num_classes(args)
        num_tasks = get_num_tasks(args)
        self.input_size = input_size_match[args.data]
        # Set an empty to store the indexes for the selected exemplars
        self.alpha_dr_herding = np.zeros((num_tasks, self.dictionary_size, self.nb_cl_per_task), np.float32)
        # Directly load the tensors for the training samples
        self.prototypes = np.zeros((num_classes, self.dictionary_size, *self.input_size))

    def train_epoch(self, dataloader, epoch):
        # ToDo: if using NCM classifier, need to revise this function.

        total = 0
        correct = 0
        epoch_loss = 0
            
        self.model.train()

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)

            if y.size == 1:
                y.unsqueeze()

            self.optimizer.zero_grad()
            loss_ce = 0

            if self.task_now > 0:  # Replay after 1st task
                x_buf, y_buf = self.buffer.retrieve(x=x, y=y)
                outputs_buf = self.model(x_buf)
                loss_ce = self.criterion(outputs_buf, y_buf)

            outputs = self.model(x)
            loss_ce += self.criterion(outputs, y)
            loss_ce.backward()
            self.optimizer_step(epoch=epoch)

            if self.er_mode == 'online':
                self.buffer.update(x, y)

            epoch_loss += loss_ce
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)

        return epoch_loss, epoch_acc

    def after_task(self, x_train, y_train):
        self.learned_classes += self.classes_in_task
        self.model.load_state_dict(torch.load(self.ckpt_path))

        # Exemplars
        self.X_protoset_cumuls = []
        self.Y_protoset_cumuls = []
        #  mean feature of each class, dim=0 is calculated with exemplars and dim=1 is with all train data (ideal)
        self.class_means = np.zeros((self.args.feature_dim, get_num_classes(self.args), 2))

        #  Get the index of selected samples, use all mem size
        nb_protos_cl = int(np.ceil(self.buffer.mem_size / len(self.learned_classes)))  # N mem samples per class
        num_features = self.args.feature_dim
        for i in self.classes_in_task:  # classes in this task
            X_i, _ = extract_samples_according_to_labels(x_train, y_train, target_ids=[i])
            num_samples = X_i.shape[0]

            # if mem_budget per cls is larger than num_samples, set it to num_samples
            nb_protos_cl = num_samples if nb_protos_cl > num_samples else nb_protos_cl

            self.prototypes[i][:num_samples] = X_i

            Y_i = np.zeros(num_samples)
            evalloader = Dataloader_from_numpy(X_i, Y_i, batch_size=self.args.batch_size, shuffle=False)

            # Get features of that class
            mapped_prototypes = compute_features(self.model, evalloader, num_samples, num_features, self.device)
            D = mapped_prototypes.T  #(128, num_samples)
            D = D / np.linalg.norm(D, axis=0)
            mu = np.mean(D, axis=1)  # class mean  (128,)

            index1 = self.task_now  # Task
            index2 = self.classes_in_task.index(i)  # Class
            self.alpha_dr_herding[index1, :, index2] = self.alpha_dr_herding[index1, :, index2] * 0  # (Nt, Nsmp, 2)
            w_t = mu
            iter_herding = 0
            iter_herding_eff = 0

            while not np.sum(self.alpha_dr_herding[index1, :, index2] != 0) == nb_protos_cl and iter_herding_eff < 5000:
                tmp_t = np.dot(w_t, D)
                ind_max = np.argmax(tmp_t)  # winner's index
                iter_herding_eff += 1
                if self.alpha_dr_herding[index1, ind_max, index2] == 0:
                    self.alpha_dr_herding[index1, ind_max, index2] = 1 + iter_herding  # write the order
                    iter_herding += 1
                w_t = w_t + mu - D[:, ind_max]  # remove the winner

            # If not all the alpha are selected before 5000 iteration, randomly select remaining exemplars.
            # Without this, number of exemplars of different cls may be different and causes bug.
            alph = self.alpha_dr_herding[index1, :, index2]
            alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.  # 0. or 1. Choose this sample as exemplar if == 1
            idx_selected = np.where(alph == 1)[0]
            offset = nb_protos_cl - np.where(alph == 1)[0].shape[0]
            while offset>0:
                ind_random = np.random.randint(0, num_samples)
                if ind_random in idx_selected:
                    continue
                self.alpha_dr_herding[index1, ind_random, index2] = 1 + iter_herding  # write the order
                idx_selected = np.append(idx_selected, ind_random)
                iter_herding += 1
                offset -= 1

        ############### Update exemplars #############
        for t in range(self.task_now+1):
            for cls in range(self.nb_cl_per_task):
                current_cls = int(t * self.nb_cl_per_task + cls)
                data_cls = self.prototypes[current_cls]
                num_samples = data_cls.shape[0]
                target = np.zeros(num_samples)
                evalloader = Dataloader_from_numpy(data_cls, target, batch_size=self.args.batch_size, shuffle=False)

                # Get features of that class
                self.model.eval()
                mapped_prototypes = compute_features(self.model, evalloader, num_samples, num_features, self.device)
                D = mapped_prototypes.T
                D = D / np.linalg.norm(D, axis=0)
                alph = self.alpha_dr_herding[t, :, cls]
                alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.  # 0. or 1. Choose this sample as exemplar if == 1
                self.X_protoset_cumuls.append(self.prototypes[current_cls, np.where(alph == 1)[0]])
                self.Y_protoset_cumuls.append(current_cls * np.ones(len(np.where(alph == 1)[0])))

                alph = alph / np.sum(alph)  # Mean of exemplars of that class
                self.class_means[:, current_cls, 0] = np.dot(D, alph)
                self.class_means[:, current_cls, 0] /= np.linalg.norm(self.class_means[:, current_cls, 0])
                alph = np.ones(self.dictionary_size) / self.dictionary_size  # Mean of all training samples of that class
                self.class_means[:, current_cls, 1] = np.dot(D, alph)
                self.class_means[:, current_cls, 1] /= np.linalg.norm(self.class_means[:, current_cls, 1])
        current_means = self.class_means[:, self.learned_classes]  # (128, cls_so_far, 2)

        if self.mnemonics:
            self.mnemonics_label = []
            the_X_protoset_array = np.array(self.X_protoset_cumuls[-self.nb_cl_per_task:])
            the_Y_protoset_cumuls = np.array(self.Y_protoset_cumuls[-self.nb_cl_per_task:])
            self.mnemonics_data = torch.from_numpy(the_X_protoset_array).type(torch.FloatTensor)
            self.mnemonics_label = torch.from_numpy(the_Y_protoset_cumuls)
            self.mnemonics = nn.ParameterList()
            self.mnemonics.append(nn.Parameter(self.mnemonics_data))
            device = self.device
            self.mnemonics.to(device)
            self.model.eval()
            self.mnemonics_optimizer = optim.SGD(self.mnemonics, lr=self.args.mnemonics_lr, momentum=0.9,
                                                 weight_decay=5e-4)
            current_means_new = current_means[:, :, 0].T  #  means of all cls so far calculated with exemplars

            trainloader = Dataloader_from_numpy(x_train, y_train, batch_size=self.args.batch_size, shuffle=True)

            for epoch in range(self.args.mnemonics_epochs):
                # self.mnemonics_lr_scheduler.step()
                for batch_idx, (q_inputs, q_targets) in enumerate(trainloader):  # training data of current task
                    q_inputs, q_targets = q_inputs.to(device), q_targets.to(device)
                    q_feature = self.model.feature(q_inputs)
                    self.mnemonics_optimizer.zero_grad()
                    mnemonics_outputs = self.model.feature(self.mnemonics[0][0])
                    this_class_mean_mnemonics = torch.mean(mnemonics_outputs, dim=0)  # (128, )
                    total_class_mean_mnemonics = this_class_mean_mnemonics.unsqueeze(dim=0)  # (1, 128)

                    for mnemonics_idx in range(len(self.mnemonics[0]) - 1):  # loop over 1st --> latest class
                        mnemonics_outputs = self.model.feature(self.mnemonics[0][mnemonics_idx + 1])
                        this_class_mean_mnemonics = torch.mean(mnemonics_outputs, dim=0)
                        total_class_mean_mnemonics = torch.cat(
                            (total_class_mean_mnemonics, this_class_mean_mnemonics.unsqueeze(dim=0)), dim=0)

                    if self.task_now == 0:
                        all_cls_means = total_class_mean_mnemonics  # (50, 64)
                    else:
                        all_cls_means = torch.tensor(current_means_new).float().to(device)
                        all_cls_means[-self.nb_cl_per_task:] = total_class_mean_mnemonics  # Update cls means with current task's mnemonics

                    # Classification over all classes
                    the_logits = F.linear(F.normalize(q_feature, p=2, dim=1),
                                          F.normalize(all_cls_means, p=2, dim=1))  # (128, 50)
                    loss = F.cross_entropy(the_logits, q_targets)
                    loss.backward()
                    self.mnemonics_optimizer.step()

            # Update the exemplars of current task with mnemonics
            for i in self.classes_in_task:
                mnemonics_array_new = np.array(self.mnemonics[0][self.classes_in_task.index(i)].detach().to('cpu'))
                self.X_protoset_cumuls[i] = mnemonics_array_new

            # Update prototypes with Mnemonics
            X_protoset_array = np.array(self.X_protoset_cumuls)
            X_protoset_cumuls_idx = 0
            for t in range(self.task_now + 1):
                for cls in range(self.nb_cl_per_task):
                    alph = self.alpha_dr_herding[t, :, cls]
                    alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.
                    this_X_protoset_array = X_protoset_array[X_protoset_cumuls_idx]
                    X_protoset_cumuls_idx += 1
                    this_X_protoset_array = this_X_protoset_array.astype(np.float64)
                    self.prototypes[t * self.nb_cl_per_task + cls, np.where(alph == 1)[0]] = this_X_protoset_array

        X_protoset = np.concatenate(self.X_protoset_cumuls)  # X_protoset_cumuls is a list
        Y_protoset = np.concatenate(self.Y_protoset_cumuls)
        self.buffer.buffer_input = torch.FloatTensor(X_protoset).to(self.device)
        self.buffer.buffer_label = torch.LongTensor(Y_protoset).to(self.device)
        self.buffer.current_index = self.buffer.buffer_input.size(0)  # Totally filled the buffer

        self.means_of_exemplars = compute_cls_feature_mean_buffer(self.buffer, self.model)

