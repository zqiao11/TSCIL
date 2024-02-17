import torch
from agents.base import BaseLearner
from utils.buffer.buffer import Buffer
from utils.data import Dataloader_from_numpy
import copy
import torch.nn.functional as F
from utils.setup_elements import get_num_classes
from agents.utils.functions import compute_cls_feature_mean_buffer


class DarkExperienceReplay(BaseLearner):
    """
    Follow the Paper: "Dark Experience for General Continual Learning: a Strong, Simple Baseline"
    https://github.com/aimagelab/mammoth
    """
    def __init__(self, model, args):
        super(DarkExperienceReplay, self).__init__(model, args)
        args.eps_mem_batch = args.batch_size
        args.retrieve = 'random'
        args.update = 'random'
        self.buffer = Buffer(model, args)
        self.ncm_classifier = args.ncm_classifier
        print('ER mode: {}, NCM classifier: {}'.format(self.er_mode, self.ncm_classifier))

        self.der_plus = args.der_plus

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
                x_buf, (y_buf, logits_buf) = self.buffer.retrieve(x=x, y=y)
                outputs_buf = self.model(x_buf)
                loss += 0.5 * F.mse_loss(outputs_buf, logits_buf[:, :outputs_buf.size(1)])

                if self.der_plus:
                    loss += 0.5 * self.criterion(outputs_buf, y_buf)

            outputs = self.model(x)
            loss += self.criterion(outputs, y)
            loss.backward()
            self.optimizer_step(epoch=epoch)

            if self.er_mode == 'online':
                self.buffer.update(x, y, logits=outputs)

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
            dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
            for batch_id, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    outputs = self.model(x)
                bs, n_cls = outputs.size()
                zeros_tensor = torch.zeros((bs, get_num_classes(self.args) - n_cls), device=self.device)
                outputs = torch.cat((outputs, zeros_tensor), dim=1)

                self.buffer.update(x, y, logits=outputs)

        # Compute means of classes if using ncm classifier
        if self.ncm_classifier:
            self.means_of_exemplars = compute_cls_feature_mean_buffer(self.buffer, self.model)

        if self.use_kd:
            self.teacher = copy.deepcopy(self.model)  # eval()
            if not self.args.teacher_eval:
                self.teacher.train()