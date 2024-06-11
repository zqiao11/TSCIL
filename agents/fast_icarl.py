import torch
from agents.base import BaseLearner
from utils.buffer.buffer import Buffer
import heapq
from utils.data import extract_samples_according_to_labels, Dataloader_from_numpy
from agents.utils.functions import compute_cls_feature_mean_buffer
import numpy as np


class FastICARL(BaseLearner):
    """
    Follow the Paper: "FastICARL: Fast Incremental Classifier and Representation Learning with Efficient Budget Allocation in Audio Sensing Applications".
    """

    def __init__(self, model, args):
        super(FastICARL, self).__init__(model, args)
        args.eps_mem_batch = args.batch_size
        args.retrieve = 'random'
        args.update = 'random'  # Set as 'random' to initialize buffer
        self.buffer = Buffer(model, args)
        self.ncm_classifier = True
        self.args.criterion = 'BCE'
        print('ER mode: {}, NCM classifier: {}'.format(self.er_mode, self.ncm_classifier))

    def train_epoch(self, dataloader, epoch):
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
            loss = 0

            if self.task_now > 0:  # Replay after 1st task
                x_buf, y_buf = self.buffer.retrieve(x=x, y=y)
                outputs_buf = self.model(x_buf)
                loss = self.criterion(outputs_buf, y_buf)

            outputs = self.model(x)
            loss += self.criterion(outputs, y)
            loss.backward()
            self.optimizer_step(epoch=epoch)

            if self.er_mode == 'online':
                self.buffer.update(x, y)

            epoch_loss += loss
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def after_task(self, x_train, y_train):
        self.learned_classes += self.classes_in_task
        self.model.load_state_dict(torch.load(self.ckpt_path))
        nb_protos_cl = int(np.ceil(self.buffer.mem_size / len(self.learned_classes)))
        X_protoset_cumuls = []
        Y_protoset_cumuls = []

        for i in self.classes_in_task:
            X_i, Y_i = extract_samples_according_to_labels(x_train, y_train, target_ids=[i])
            dataloader = Dataloader_from_numpy(X_i, Y_i, batch_size=self.args.batch_size, shuffle=False)
            features = []
            for (x, y) in dataloader:
                x = x.to(self.device)
                features.append(self.model.feature(x).detach())
            features = torch.cat(features)
            mu = features.mean(0)
            distances = (features - mu).norm(dim=1)

            # Create a max heap with size m
            heap = []
            for i, dist in enumerate(distances):
                if len(heap) < nb_protos_cl:
                    # We use negative distance because heapq in python is a min-heap.
                    # By using negative, we are effectively creating a max heap.
                    heapq.heappush(heap, (-dist.item(), i))
                else:
                    if dist < -heap[0][0]:
                        heapq.heappop(heap)  # pop the largest distance (most negative value in heap)
                        heapq.heappush(heap, (-dist.item(), i))

            # Now, extract the indices in descending order of their distance
            exemplar_indices = [heapq.heappop(heap)[1] for _ in range(len(heap))]  # order: farthest to closest
            exemplar_indices.reverse()  # reverse to get the correct order: closest to farthest

            # Select the exemplars and quantize them
            exemplars = X_i[exemplar_indices]
            labels = Y_i[exemplar_indices]
            X_protoset_cumuls.append(exemplars)
            Y_protoset_cumuls.append(labels)
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
        self.means_of_exemplars = compute_cls_feature_mean_buffer(self.buffer, self.model)


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