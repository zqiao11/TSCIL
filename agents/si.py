import numpy as np
import torch
from torch.nn.modules.batchnorm import _NormBase
from agents.base import BaseLearner
from utils.data import Dataloader_from_numpy
from utils.utils import EarlyStopping
from fnmatch import fnmatch
from torch.optim import lr_scheduler
from utils.optimizer import adjust_learning_rate

class SI(BaseLearner):
    """
    https://avalanche-api.continualai.org/en/v0.2.0/_modules/avalanche/training/plugins/synaptic_intelligence.html#SynapticIntelligencePlugin
    """
    def __init__(self, model, args):
        super(SI, self).__init__(model, args)

        self.si_lambda = (
            args.lambda_impt if isinstance(args.lambda_impt, (list, tuple)) else [args.lambda_impt]
        )  # list of lambda

        self.eps: float = 0.0000001
        self.excluded_parameters = set()
        self.ewc_data = (dict(), dict())

        """
        The first dictionary contains the args at loss minimum while the 
        second one contains the parameter importance.
        """

        self.syn_data = {
            "old_theta": dict(),
            "new_theta": dict(),
            "grad": dict(),
            "trajectory": dict(),
            "cum_trajectory": dict(),
        }

    def learn_task(self, task):

        (x_train, y_train), (x_val, y_val), _ = task

        self.before_task(y_train)

        SI.create_syn_data(
            self.model,
            self.ewc_data,
            self.syn_data,
            self.excluded_parameters,
        )  # Initialize syn_data of this task

        SI.init_batch(
            self.model,
            self.ewc_data,
            self.syn_data,
            self.excluded_parameters,
        )  # Copy the importance calculated before

        train_dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
        val_dataloader = Dataloader_from_numpy(x_val, y_val, self.batch_size, shuffle=False)
        early_stopping = EarlyStopping(path=self.ckpt_path, patience=self.args.patience, mode='min', verbose=False)

        self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                                                 steps_per_epoch=len(train_dataloader),
                                                 epochs=self.epochs,
                                                 max_lr=self.args.lr)

        for epoch in range(self.epochs):
            epoch_loss_train, epoch_acc_train, epoch_loss_si = self.train_epoch(train_dataloader, epoch=epoch)
            epoch_loss_val, epoch_acc_val = self.cross_entropy_epoch_run(val_dataloader, mode='val')

            if self.args.lradj != 'TST':
                adjust_learning_rate(self.optimizer, self.scheduler, epoch + 1, self.args)

            if self.verbose:
                print('Epoch {}/{}: Loss = {}, Avg_Syn_term = {}, Accuracy = {}'.format(epoch + 1, self.epochs,
                                                                                           epoch_loss_train,
                                                                                           epoch_loss_si,
                                                                                           epoch_acc_train))

            early_stopping(epoch_loss_val, self.model)
            if early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                break

        self.after_task()

    def train_epoch(self, dataloader, epoch):
        total = 0
        correct = 0
        epoch_loss = 0
        epoch_syn_loss = 0

        self.model.train()

        for batch_id, (x, y) in enumerate(dataloader):

            SI.pre_update(
                self.model, self.syn_data, self.excluded_parameters
            )  # Update the 'old_theta' (model args before this iteration)

            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)

            self.optimizer.zero_grad()
            outputs = self.model(x)

            if self.task_now == 0:
                step_loss = self.criterion(outputs, y.long())
                syn_loss = 0
            else:
                try:
                    si_lamb = self.si_lambda[self.task_now]
                except IndexError:  # less than one lambda per experience, take last
                    si_lamb = self.si_lambda[-1]

                syn_loss = SI.compute_ewc_loss(
                    self.model,
                    self.ewc_data,
                    self.excluded_parameters,
                    lambd=si_lamb,
                    device=self.device,
                )

                step_loss = self.criterion(outputs, y.long()) + syn_loss

            step_loss.backward()
            self.optimizer_step(epoch=epoch)

            SI.post_update(
                self.model, self.syn_data, self.excluded_parameters
            )  # Update 'new_theta' and 'grad' after this iteration, calculate trajectory.

            epoch_loss += step_loss
            epoch_syn_loss += syn_loss
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)
        epoch_syn_loss /= (batch_id + 1)

        return epoch_loss, epoch_acc, epoch_syn_loss

    def after_task(self, x_train=None, y_train=None):
        """
        Calculate Fisher
        :return:
        """
        self.learned_classes += self.classes_in_task
        self.model.load_state_dict(torch.load(self.ckpt_path))

        SI.update_ewc_data(
            self.model,
            self.ewc_data,
            self.syn_data,
            0.001,
            self.excluded_parameters,
            1,
            eps=self.eps,
        )

    @staticmethod
    @torch.no_grad()
    def create_syn_data(model, ewc_data, syn_data, excluded_parameters):
        params = SI.allowed_parameters(
            model, excluded_parameters
        )

        for param_name, param in params:
            if param_name not in ewc_data[0]:
                # Handles added architecture, like new head (doesn't manage parameter expansion!)
                ewc_data[0][param_name] = SI._zero(param)
                ewc_data[1][param_name] = SI._zero(param)

                syn_data["old_theta"][param_name] = SI._zero(param)
                syn_data["new_theta"][param_name] = SI._zero(param)
                syn_data["grad"][param_name] = SI._zero(param)
                syn_data["trajectory"][param_name] = SI._zero(param)
                syn_data["cum_trajectory"][param_name] = SI._zero(param)

            # For old args without expansion
            if param.flatten().shape == ewc_data[0][param_name].shape:
                continue

            # For old args with expansion, eg. single head with new neurons
            else:
                # Update the size of the param
                syn_data["old_theta"][param_name] = SI._zero(param)
                syn_data["new_theta"][param_name] = SI._zero(param)
                syn_data["grad"][param_name] = SI._zero(param)
                syn_data["trajectory"][param_name] = SI._zero(param)

                # Copy the reserved values
                with torch.no_grad():
                    old_ewc_data0 = ewc_data[0][param_name]
                    ewc_data[0][param_name] = SI._zero(param)
                    ewc_data[0][param_name][:old_ewc_data0.shape[0]] = old_ewc_data0

                    old_ewc_data1 = ewc_data[1][param_name]
                    ewc_data[1][param_name] = SI._zero(param)
                    ewc_data[1][param_name][:old_ewc_data1.shape[0]] = old_ewc_data1

                    old_cum_trajectory = syn_data["cum_trajectory"][param_name]
                    syn_data["cum_trajectory"][param_name] = SI._zero(param)
                    syn_data["cum_trajectory"][param_name][:old_cum_trajectory.shape[0]] = old_cum_trajectory


    @staticmethod
    @torch.no_grad()
    def _zero(param):
        return torch.zeros(param.numel(), dtype=param.dtype)

    @staticmethod
    @torch.no_grad()
    def extract_weights(model, target, excluded_parameters):
        params = SI.allowed_parameters(
            model, excluded_parameters
        )

        for name, param in params:
            target[name][...] = param.detach().cpu().flatten()

    @staticmethod
    @torch.no_grad()
    def extract_grad(model, target, excluded_parameters):
        params = SI.allowed_parameters(
            model, excluded_parameters
        )

        # Store the gradients into target
        for name, param in params:
            target[name][...] = param.grad.detach().cpu().flatten()

    @staticmethod
    @torch.no_grad()
    def init_batch(model, ewc_data, syn_data, excluded_parameters):
        # Keep initial importance weights?
        SI.extract_weights(
            model, ewc_data[0], excluded_parameters
        )
        # Refresh trajectory
        for param_name, param_trajectory in syn_data["trajectory"].items():
            param_trajectory.fill_(0.0)

    @staticmethod
    @torch.no_grad()
    def pre_update(model, syn_data, excluded_parameters):
        SI.extract_weights(
            model, syn_data["old_theta"], excluded_parameters
        )

    @staticmethod
    @torch.no_grad()
    def post_update(model, syn_data, excluded_parameters):
        SI.extract_weights(
            model, syn_data["new_theta"], excluded_parameters
        )
        SI.extract_grad(
            model, syn_data["grad"], excluded_parameters
        )

        for param_name in syn_data["trajectory"]:
            syn_data["trajectory"][param_name] += syn_data["grad"][
                                                      param_name
                                                  ] * (
                                                          syn_data["new_theta"][param_name]
                                                          - syn_data["old_theta"][param_name]
                                                  )

    @staticmethod
    def compute_ewc_loss(model, ewc_data, excluded_parameters, device, lambd=0.0):
        params = SI.allowed_parameters(
            model, excluded_parameters
        )

        loss = None
        for name, param in params:
            weights = param.to(device).flatten()  # Flat, not detached
            param_ewc_data_0 = ewc_data[0][name].to(device)  # Flat, detached
            param_ewc_data_1 = ewc_data[1][name].to(device)  # Flat, detached

            syn_loss = torch.dot(
                param_ewc_data_1, (weights - param_ewc_data_0) ** 2
            ) * (lambd / 2)

            if loss is None:
                loss = syn_loss
            else:
                loss += syn_loss

        return loss

    @staticmethod
    @torch.no_grad()
    def update_ewc_data(net, ewc_data, syn_data, clip_to, excluded_parameters, c=0.0015, eps: float = 0.0000001):
        SI.extract_weights(
            net, syn_data["new_theta"], excluded_parameters
        )

        for param_name in syn_data["cum_trajectory"]:
            syn_data["cum_trajectory"][param_name] += (
                    c
                    * syn_data["trajectory"][param_name]
                    / (
                            np.square(
                                syn_data["new_theta"][param_name]
                                - ewc_data[0][param_name]
                            )
                            + eps
                    )
            )

        for param_name in syn_data["cum_trajectory"]:
            ewc_data[1][param_name] = torch.empty_like(
                syn_data["cum_trajectory"][param_name]
            ).copy_(-syn_data["cum_trajectory"][param_name])

        # change sign here because the Ewc regularization
        # in Caffe (theta - thetaold) is inverted w.r.t. syn equation [4]
        # (thetaold - theta)
        for param_name in ewc_data[1]:
            ewc_data[1][param_name] = torch.clamp(
                ewc_data[1][param_name], max=clip_to
            )
            ewc_data[0][param_name] = syn_data["new_theta"][param_name].clone()

    @staticmethod
    def explode_excluded_parameters(excluded):
        """
        Explodes a list of excluded parameters by adding a generic final ".*"
        wildcard at its end.

        :param excluded: The original set of excluded parameters.

        :return: The set of excluded parameters in which ".*" patterns have been
            added.
        """
        result = set()
        for x in excluded:
            result.add(x)
            if not x.endswith("*"):
                result.add(x + ".*")
        return result

    @staticmethod
    def not_excluded_parameters(model, excluded_parameters):
        # Add wildcards ".*" to all excluded parameter names
        result = []
        excluded_parameters = (
            SI.explode_excluded_parameters(
                excluded_parameters
            )
        )
        layers_params = get_layers_and_params(model)

        for lp in layers_params:
            if isinstance(lp[1], _NormBase):
                # Exclude batch norm parameters
                excluded_parameters.add(lp[2])

        for name, param in model.named_parameters():
            accepted = True
            for exclusion_pattern in excluded_parameters:
                if fnmatch(name, exclusion_pattern):
                    accepted = False
                    break

            if accepted:
                result.append((name, param))

        return result

    @staticmethod
    def allowed_parameters(model, excluded_parameters):

        allow_list = SI.not_excluded_parameters(
            model, excluded_parameters
        )

        result = []
        for name, param in allow_list:
            if param.requires_grad:
                result.append((name, param))

        return result


def get_layers_and_params(model, prefix=''):
    result = []
    for param_name, param in model.named_parameters(recurse=False):
        result.append((prefix[:-1], model, prefix + param_name, param))

    for layer_name, layer in model.named_modules():
        if layer == model:
            continue

        layer_complete_name = prefix + layer_name + '.'

        result += get_layers_and_params(layer, prefix=layer_complete_name)

    return result