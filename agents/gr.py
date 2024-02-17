import torch
from agents.base import BaseLearner
from torch.optim import Adam
import copy
from utils.setup_elements import input_size_match
from models.timeVAE import VariationalAutoencoderConv
from utils.data import Dataloader_from_numpy
from utils.utils import EarlyStopping
from torch.optim import lr_scheduler
from utils.optimizer import adjust_learning_rate


class GenerativeReplay(BaseLearner):
    """
    Continual learning with deep generative replay, NIPS 2017
    Code reference: https://github.com/GMvandeVen/continual-learning/tree/master
    TimeVAE: https://github.com/abudesai/timeVAE
    """
    def __init__(self, model, args):
        super(GenerativeReplay, self).__init__(model, args)
        input_size = input_size_match[args.data]

        self.batch_size = args.batch_size
        self.generator = VariationalAutoencoderConv(seq_len=input_size[0],
                                                    feat_dim=input_size[1],
                                                    latent_dim=args.feature_dim,  # 2 for visualization
                                                    hidden_layer_sizes=[64, 128, 256, 512],  # [128, 256]
                                                    device=self.device,
                                                    recon_wt=args.recon_wt)

        self.epochs_g = args.epochs_g
        self.optimizer_g = Adam(self.generator.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
        self.ckpt_path_g = self.ckpt_path.replace('/ckpt', '/vae_ckpt')

        self.previous_generator = None
        self.previous_model = None

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
        # Train the learner
        if self.verbose:
            print("Training the learner...")

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

        # Train the generator
        if self.verbose:
            print("Training the generator...")

        early_stopping = EarlyStopping(path=self.ckpt_path_g, patience=20, mode='min', verbose=False)
        for epoch in range(self.epochs_g):
            for batch_id, (x, y) in enumerate(train_dataloader):
                x = x.to(self.device)

                if self.task_now > 0:  # Replay after 1st task
                    x_ = self.previous_generator.sample(self.batch_size)
                else:
                    x_ = None

                # generator's input should be (N, L, C)
                rnt = 1 / (self.task_now + 1) if self.args.adaptive_weight else 0.5
                generator_loss_dict = self.generator.train_a_batch(x=x.transpose(1, 2),
                                                                   optimizer=self.optimizer_g,
                                                                   x_=x_,
                                                                   rnt=rnt)

            train_mse_loss, train_kl_loss = self.generator.evaluate(train_dataloader)
            # Validate on val set for early stop
            val_mse_loss, val_kl_loss = self.generator.evaluate(val_dataloader)
            if self.verbose:
                print('Epoch {}/{}: Recons Loss = {}, KL Divergence = {}'.format(epoch + 1, self.epochs_g,
                                                                                 train_mse_loss, train_kl_loss))

            early_stopping(val_mse_loss, self.generator)
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
            x, y = x.to(self.device), y.to(self.device)  # x is in shape of (N, L, C)
            total += y.size(0)

            if y.size == 1:
                y.unsqueeze()

            self.optimizer.zero_grad()
            loss_ce = 0

            if self.task_now > 0:  # Generative Replay after 1st task
                x_ = self.previous_generator.sample(self.batch_size)  # x_ is in shape of (N, C, L)

                with torch.no_grad():
                    all_scores_ = self.previous_model(x_.transpose(1, 2))  # model's input should be (N, L, C)
                    _, y_ = torch.max(all_scores_, dim=1)

                # Train the classifier model on this batch
                outputs_ = self.model(x_.transpose(1, 2))
                loss_ce = self.criterion(outputs_, y_)

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
        super(GenerativeReplay, self).after_task(x_train, y_train)
        self.generator.load_state_dict(torch.load(self.ckpt_path_g))  # eval
        self.generator.reset_trackers()
        self.previous_generator = copy.deepcopy(self.generator).eval()
        self.previous_model = copy.deepcopy(self.model).eval()
