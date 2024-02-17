import copy
import numpy as np
import torch
from agents.base import BaseLearner
from utils.data import Dataloader_from_numpy, Dataloader_from_numpy_with_idx
from utils.utils import EarlyStopping
from torch.optim import lr_scheduler
from utils.optimizer import adjust_learning_rate
from utils.buffer.buffer import Buffer
from scipy.stats import entropy
from scipy.special import softmax
from torch import nn


class CLOPS(BaseLearner):
    """
    Follow the CLOPS Paper: https://www.nature.com/articles/s41467-021-24483-0
    Importance-guided buffer storage
    Uncertainty-based buffer retrieval.
    """
    def __init__(self, model, args):
        super(CLOPS, self).__init__(model, args)

        args.eps_mem_batch = args.batch_size
        # We need to set args.retrieve and args.update as 'random' to initialize Buffer
        args.retrieve = 'random'
        args.update = 'random'
        self.buffer = Buffer(model, args)
        self.er_mode = args.er_mode
        print('ER mode: {}'.format(self.er_mode))

        self.instance_beta = None
        self.beta_lr = args.beta_lr
        self.lambda_beta = args.lambda_beta
        self.tracked_instance_beta_dict = None
        self.track_step = 1  # step for tracking beta

        # Use MC-dropout uncertainty retrieve as paper or not. If not, use random.
        self.mc_retrieve = args.mc_retrieve
        self.mc_epochs = 3  # Interval of epochs to trigger MC sampling
        self.mc_times = 10  # How many times to repeat


    def learn_task(self, task):

        (x_train, y_train), (x_val, y_val), _ = task

        # Prepare for beta
        nsamples = x_train.shape[0]
        instance_idx = np.arange(nsamples)  # Get the idx of each training instance
        self.instance_beta = torch.ones(nsamples, requires_grad=False, device=self.device)  # sample-wise coefficient
        self.tracked_instance_beta_dict = {index: [] for index in range(nsamples)}

        self.before_task(y_train)
        train_dataloader = Dataloader_from_numpy_with_idx(x_train, instance_idx, y_train, self.batch_size, shuffle=True)
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
                self.epoch_loss_printer(epoch, epoch_acc_train, epoch_loss_train)

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

        criterion_buf = torch.nn.CrossEntropyLoss()
        criterion_new = torch.nn.CrossEntropyLoss(reduction='none')  # weight the loss per instance
        regularization_criterion = torch.nn.MSELoss()
        self.model.train()

        for batch_id, (x, idx, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)

            if y.size == 1:
                y.unsqueeze()

            self.optimizer.zero_grad()
            loss = 0

            if self.task_now > 0:  # Replay after 1st task
                if self.mc_retrieve:
                    mc_sampling = True if epoch % self.mc_epochs == 0 else False
                    x_buf, y_buf = self.uncertainty_retrieve(mc_sampling)
                else:
                    x_buf, y_buf = self.buffer.retrieve(x=x, y=y)
                outputs_buf = self.model(x_buf)
                loss = criterion_buf(outputs_buf, y_buf)

            outputs = self.model(x)
            ce_samplewise = criterion_new(outputs, y)

            # Create a new tensor for the beta values corresponding to each sample in the batch
            beta_batch = torch.tensor([self.instance_beta[i].item() for i in idx], device=self.device,
                                      requires_grad=True)

            # Perform the multiplication and mean reduction
            ce_weighted = torch.mean(ce_samplewise * beta_batch)
            loss = loss + ce_weighted

            # Add regularization term
            regularization_loss = regularization_criterion(beta_batch, torch.ones_like(beta_batch))
            loss += self.lambda_beta * regularization_loss
            loss.backward()

            # Update the beta values in self.instance_beta
            with torch.no_grad():
                for i, index in enumerate(idx):
                    self.instance_beta[index] -= self.beta_lr * beta_batch.grad[i]

            self.optimizer.step()

            # Track the beta values
            if epoch % self.track_step == 0:
                for i in range(x.shape[0]):
                    instance_id = idx[i]
                    instance_id = instance_id.item()
                    beta_i = self.instance_beta[instance_id].cpu().detach().item()
                    self.tracked_instance_beta_dict[instance_id].append(beta_i)

            epoch_loss += loss
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)

        return epoch_loss, epoch_acc

    def after_task(self, x_train, y_train):
        self.learned_classes += self.classes_in_task
        self.model.load_state_dict(torch.load(self.ckpt_path))

        if self.buffer and self.er_mode == 'task':  # Additional pass to collect memory samples
            nb_protos_cl = int(np.ceil(self.buffer.mem_size / len(self.learned_classes)))
            X_protoset_cumuls = []
            Y_protoset_cumuls = []

            # Save the top-k samples per cls
            for cls in self.classes_in_task:
                idx_cls = np.where(y_train == cls)[0]
                # Calculate the area under beta's line for each instance
                aul_dict = dict()
                subset = {key: self.tracked_instance_beta_dict[key] for key in list(idx_cls)}
                for index, beta_over_time in subset.items():
                    mean_alpha = np.trapz(beta_over_time)  # Eq 7
                    aul_dict[index] = mean_alpha
                sorted_aul_dict = dict(sorted(aul_dict.items(), key=lambda x: x[1], reverse=True))
                buffered_indices = list(sorted_aul_dict.keys())[:nb_protos_cl]
                X_protoset_cumuls.append(x_train[buffered_indices])
                Y_protoset_cumuls.append(y_train[buffered_indices])

            X_protoset = np.concatenate(X_protoset_cumuls)
            Y_protoset = np.concatenate(Y_protoset_cumuls)
            X_protoset = torch.FloatTensor(X_protoset).to(self.device)
            Y_protoset = torch.LongTensor(Y_protoset).to(self.device)

            # extract old exemplars and delete part of them
            if self.task_now > 0:
                kept_exemplars, kept_labels = self.remove_old_exemplars(nb_protos_cl)
                X_protoset = torch.cat((kept_exemplars, X_protoset))
                Y_protoset = torch.cat((kept_labels, Y_protoset))

            self.buffer.buffer_input = X_protoset
            self.buffer.buffer_label = Y_protoset
            self.buffer.current_index = self.buffer.buffer_input.size(0)  # Totally filled the buffer

        if self.use_kd:
            self.teacher = copy.deepcopy(self.model)  # eval()
            if not self.args.teacher_eval:
                self.teacher.train()


    def remove_old_exemplars(self, n_exm_per_task):
        old_exemplars = self.buffer.buffer_input
        old_labels = self.buffer.buffer_label  # Tensors now
        num_old_cls = len(list(set(self.learned_classes) - set(self.classes_in_task)))
        num_exm_per_old_cls = int(np.ceil(self.buffer.mem_size / num_old_cls))

        kept_exemplars = []
        kept_labels = []

        for i in range(num_old_cls):
            start = i * num_exm_per_old_cls
            exem_i = old_exemplars[start: start + n_exm_per_task]
            labels_i = old_labels[start: start + n_exm_per_task]
            kept_exemplars.append(exem_i)
            kept_labels.append(labels_i)

        kept_exemplars = torch.cat(kept_exemplars)
        kept_labels = torch.cat(kept_labels)

        return kept_exemplars, kept_labels

    @torch.no_grad()
    def uncertainty_retrieve(self, mc_sampling):
        """
        Apply MC-dropout to collect the posterior of the memory samples.
        Return G: (N, T, C), where N is # of mem samples. T is # of MC trails. C is # of classes

        # https://github.com/danikiyasseh/CLOPS/blob/master/prepare_acquisition_functions.py#L92
        """

        if mc_sampling:
            # Activate dropout for MC-dropout
            if self.args.dropout == 0:
                for m in self.model.modules():
                    if isinstance(m, nn.Dropout):
                        m.p = 0.25

            X_retrieve = []
            Y_retrieve = []
            X_protoset = self.buffer.buffer_input
            Y_protoset = self.buffer.buffer_label
            Y_protoset_array = Y_protoset.cpu().detach().numpy()
            bs = self.args.batch_size
            n_retrieve_per_cls = int(np.ceil(bs / len(self.learned_classes)))

            N = X_protoset.size(0)
            T = self.mc_times
            C = len(self.learned_classes) + len(self.classes_in_task)  # num of nodes in classifier
            G = torch.zeros((N, T, C), device=self.device)

            # MC dropout sampling
            for t in range(T):
                start = 0
                end = bs if bs < N else N
                while start < end:
                    x = X_protoset[start: end]
                    outputs = self.model(x)
                    G[start: end, t, :] = outputs
                    start = end
                    end = start + bs if start + bs < N else N
            G = G.cpu().detach().numpy()  # nparray

            # Select the top-k samples per class for retrieval
            for cls in self.learned_classes:
                idx_cls = np.where(Y_protoset_array == cls)[0]
                bald_dict = {}
                # Compute bald_dict values for each mem sample
                for i in idx_cls:
                    array = G[i]  # TxC
                    posterior_dist = np.mean(array, 0)  # 1xC
                    posterior_dist = np.float64(posterior_dist)
                    entropy_of_mixture = retrieve_entropy(posterior_dist)
                    mixture_of_entropy = []
                    for mc_array in array:
                        entropy_of_mc = retrieve_entropy(mc_array)  # 1xC argument
                        mixture_of_entropy.append(entropy_of_mc)
                    mixture_of_entropy = np.mean(mixture_of_entropy)
                    bald = entropy_of_mixture - mixture_of_entropy
                    bald_dict[i] = bald

                sorted_bald_dict = dict(sorted(bald_dict.items(), key=lambda x: x[1], reverse=True))
                buffered_indices = list(sorted_bald_dict.keys())[:n_retrieve_per_cls]
                X_retrieve.append(X_protoset[buffered_indices])
                Y_retrieve.append(Y_protoset[buffered_indices])

            self.x_buf = torch.cat(X_retrieve).to(self.device)
            self.y_buf = torch.cat(Y_retrieve).to(self.device)

            # Set dropout back to 0
            if self.args.dropout == 0:
                for m in self.model.modules():
                    if isinstance(m, nn.Dropout):
                        m.p = 0

        return (self.x_buf, self.y_buf)


def retrieve_entropy(array): #array is 1xC
    array = softmax(array)
    entropy_estimate = entropy(array)  # entropy also accepts logit values (it will normalize it)
    return entropy_estimate
