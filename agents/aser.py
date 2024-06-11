import torch
from agents.base import BaseLearner
from utils.buffer.buffer import Buffer


class ASER(BaseLearner):
    """
    Online Class-Incremental Continual Learning with Adversarial Shapley Value, AAAI 2021
    """
    def __init__(self, model, args):
        super(ASER, self).__init__(model, args)
        args.eps_mem_batch = args.batch_size
        args.retrieve = 'ASER'
        args.update = 'ASER'
        self.buffer = Buffer(model, args)
        self.ncm_classifier = args.ncm_classifier
        print('ER mode: {}, NCM classifier: {}'.format(self.er_mode, self.ncm_classifier))

    def train_epoch(self, dataloader, epoch):
        total = 0
        correct = 0
        epoch_loss = 0

        self.model.train()

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)

            if y.size == 1:
                y.unsqueeze()

            if self.task_now == 0:
                total += y.size(0)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
            else:  # https://github.com/RaptorMai/online-continual-learning/blob/main/agents/exp_replay.py#L79
                x_buf, y_buf = self.buffer.retrieve(x=x, y=y)
                self.optimizer.zero_grad()
                combined_batch = torch.cat((x_buf, x))
                combined_labels = torch.cat((y_buf, y))
                total += combined_labels.size(0)
                outputs = self.model(combined_batch)
                loss = self.criterion(outputs, combined_labels)
                loss.backward()

            self.optimizer_step(epoch=epoch)

            if self.er_mode == 'online':
                self.buffer.update(x, y)

            epoch_loss += loss
            prediction = torch.argmax(outputs, dim=1)
            if self.task_now == 0:
                correct += prediction.eq(y).sum().item()
            else:
                correct += prediction.eq(combined_labels).sum().item()

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)

        return epoch_loss, epoch_acc