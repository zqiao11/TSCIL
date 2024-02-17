import numpy as np
import torch
import copy
from utils.optimizer import adjust_learning_rate


# ######################## Training functions for agent.model ######################################
def epoch_run(model, dataloader, opt, scheduler, criterion, epoch, args, train=True):
    """
    Train / eval with criterion.
    :param dataloader: dataloader for train/test
    :param train: set True for training, False for eval
    :return: Average loss and average accuracy on the epoch
    """
    total = 0
    correct = 0
    epoch_loss = 0

    if train:
        model.train()
    else:
        model.eval()

    for batch_id, (x, y) in enumerate(dataloader):
        x, y = x.to(args.device), y.to(args.device)
        total += y.size(0)

        if y.size == 1:
            y.unsqueeze()

        if train:
            opt.zero_grad()
            outputs = model(x)
            step_loss = criterion(outputs, y)
            step_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            opt.step()

            # BP for one step:
            if args.norm == 'BIN':
                bin_gates = [p for p in model.parameters() if getattr(p, 'bin_gate', False)]
                for p in bin_gates:
                    p.data.clamp_(min=0, max=1)

            if args.lradj == 'TST':
                adjust_learning_rate(opt, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        else:
            with torch.no_grad():
                outputs = model(x)
                step_loss = criterion(outputs, y)
        epoch_loss += step_loss
        prediction = torch.argmax(outputs, dim=1)
        correct += prediction.eq(y).sum().item()

    epoch_acc = 100. * (correct / total)
    epoch_loss /= (batch_id+1)

    return epoch_loss, epoch_acc



def test_epoch_for_cf_matrix(model, dataloader, criterion,  device='cuda'):

    total = 0
    correct = 0
    epoch_loss = 0
    y_true, y_pred = [], []
    model.eval()

    for batch_id, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        total += y.size(0)

        if y.size == 1:
            y.unsqueeze()

        with torch.no_grad():
            outputs = model(x)
            step_loss = criterion(outputs, y)

            output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            labels = y.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

        epoch_loss += step_loss
        prediction = torch.argmax(outputs, dim=1)
        correct += prediction.eq(y).sum().item()

    epoch_acc = 100. * (correct / total)
    epoch_loss /= (batch_id+1)

    return epoch_loss, epoch_acc, y_pred, y_true


@torch.no_grad()
def compute_single_cls_feature_mean(X, Y, cls_idx, model):
    Y = np.array(copy.deepcopy(Y).to('cpu'))
    indices = np.where(Y==cls_idx)[0]
    X_cls = X[indices]
    F_cls = model.feature(X_cls)  # (N, 128)
    F_cls = F_cls.T
    mu = torch.mean(F_cls, dim=1)  # class mean, (128,)

    return mu


def compute_cls_feature_mean_buffer(buffer, model):
    """
    Compute the class mean (unnormalized) using exemplars
    """
    X = buffer.buffer_input  # (mem_size, *input_size), cuda
    Y = buffer.buffer_label

    all_cls = np.array(torch.unique(Y).to('cpu'))
    all_means = []
    for cls in all_cls:
        mu = compute_single_cls_feature_mean(X, Y, cls, model)
        all_means.append(mu)
    all_means = torch.stack(all_means, dim=0)

    return all_means


def compute_features(model, evalloader, num_samples, num_features, device):
    model.eval()
    features = np.zeros([num_samples, num_features])
    start_idx = 0
    with torch.no_grad():
        for inputs, targets in evalloader:
            inputs = inputs.to(device)
            the_feature = model.feature(inputs)
            features[start_idx:start_idx + inputs.shape[0], :] = the_feature.cpu()
            start_idx = start_idx + inputs.shape[0]
    assert (start_idx == num_samples)
    return features


# ############################ Mixup ##################################################
# https://github.com/facebookresearch/mixup-cifar10/blob/eaff31ab397a90fbc0a4aac71fb5311144b3608b/train.py#L157
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ############################ dict of args ####################################
def zerolike_params_dict(model):
    """
    Create a list of (name, parameter), where parameter is initialized to zero.
    The list has as many parameters as pattern, with the same size.
    :param model: a pytorch pattern
    """

    return [
        [k, torch.zeros_like(p).to(p.device)]
        for k, p in model.named_parameters()
    ]


def copy_params_dict(model, copy_grad=False):
    """
    Create a list of (name, parameter), where parameter is copied from pattern.
    The list has as many parameters as pattern, with the same size.
    :param model: a pytorch pattern
    :param copy_grad: if True returns gradients instead of parameter values
    """

    if copy_grad:
        return [[k, p.grad.data.clone()] for k, p in model.named_parameters()]
    else:
        return [[k, p.data.clone()] for k, p in model.named_parameters()]


def euclidean_dist(fmap1, fmap2):
    """
    fmap in shape of (N, D, L)
    """
    return torch.mean(torch.linalg.norm(fmap1 - fmap2, dim=-1), dim=1)


def pod_loss_temp(F1, F2):
    # F1, F2 are in shape of (N, D, L)
    F1 = torch.sum(F1, dim=-1)  # (N, D)
    F2 = torch.sum(F2, dim=-1)
    loss = torch.linalg.norm(F1 - F2, ord=2, dim=-1)
    return loss


def pod_loss_var(F1, F2):
    # F1, F2 are in shape of (N, D, L)
    F1 = torch.sum(F1, dim=1)  # (N, L)
    F2 = torch.sum(F2, dim=1)
    loss = torch.linalg.norm(F1 - F2, ord=2, dim=-1)
    return loss