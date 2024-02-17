import torch
from agents.base import BaseLearner
from utils.buffer.buffer import Buffer
from utils.utils import EarlyStopping
from torch.optim import lr_scheduler
from utils.optimizer import adjust_learning_rate
from utils.data import Dataloader_from_numpy, Dataloader_from_numpy_with_sub
from utils.setup_elements import *
from utils.data import extract_samples_according_to_subjects
import random


class ER_on_Subject(BaseLearner):
    """
    Utilize subject information during training. Two strategies can be used:
    1. 'part': Only select memory samples from part of the subjects.
    2. 'balanced': Retrieve memory samples in a subject-balanced manner.
    """

    def __init__(self, model, args):
        super(ER_on_Subject, self).__init__(model, args)
        args.eps_mem_batch = args.batch_size
        args.retrieve = 'random'
        args.update = 'random'
        self.buffer = Buffer(model, args)
        self.ncm_classifier = args.ncm_classifier
        print('ER mode: {}, NCM classifier: {}'.format(self.er_mode, self.ncm_classifier))
        self.num_sub = n_subjects[self.args.data]
        self.sub_to_save = random.sample(list(range(self.num_sub)), self.num_sub // 2)

    def learn_task(self, task):
        """
        Basic workflow for learning a task. For particular methods, this function will be overwritten.
        """

        (x_train, y_train, sub_train), (x_val, y_val), _ = task

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

        self.after_task(x_train, y_train, sub_train)

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
            loss_ce = 0

            if self.task_now > 0:  # Replay after 1st task
                x_buf, y_buf = self.buffer.retrieve(x=x, y=y)
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

    def after_task(self, x_train, y_train, sub_train):
        self.learned_classes += self.classes_in_task
        self.model.load_state_dict(torch.load(self.ckpt_path))

        # 1. Subject-balanced sampling
        if self.args.er_sub_type == 'balanced':
            dataloader = Dataloader_from_numpy_with_sub(x_train, y_train, sub_train, self.batch_size, shuffle=True)
            for batch_id, (x, y, sub) in enumerate(dataloader):
                x, y, sub = x.to(self.device), y.to(self.device), sub.to(self.device)
                self.buffer.update(x, y, subjects=sub)

        elif self.args.er_sub_type == 'part':
            # 1. Only select memory samples from several subjects randomly
            x_sub, y_sub = extract_samples_according_to_subjects(x_train, y_train, sub_train,
                                                                 target_ids=self.sub_to_save)
            dataloader = Dataloader_from_numpy(x_sub, y_sub, self.batch_size, shuffle=True)
            for batch_id, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.buffer.update(x, y)

        else:
            raise ValueError("Incorrect ER_sub type is used")
