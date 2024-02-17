from utils.setup_elements import input_size_match
from utils import name_match
import torch
from utils.buffer.buffer_utils import BufferClassTracker
from utils.setup_elements import n_classes
from utils.setup_elements import get_buffer_size, get_num_tasks, get_num_classes


class Buffer(torch.nn.Module):
    def __init__(self, model, args, mem_size=None):
        super().__init__()
        self.params = args
        self.model = model
        self.current_index = 0
        self.n_seen_so_far = 0
        self.device = args.device

        # define buffer
        if mem_size is None:
            self.mem_size = get_buffer_size(args)
        else:
            self.mem_size = mem_size
        self.params.mem_size = self.mem_size
        self.params.num_tasks = get_num_tasks(args)
        print('buffer has %d slots' % self.mem_size)
        input_size = input_size_match[args.data]
        buffer_input = torch.FloatTensor(self.mem_size, *input_size).fill_(0).to(self.device)
        buffer_label = torch.LongTensor(self.mem_size).fill_(0).to(self.device)
        # For Dark Experience Replay, collect logits
        buffer_logits = torch.FloatTensor(self.mem_size, get_num_classes(args)).fill_(0).to(self.device) if args.agent == 'DER' else None
        buffer_sub = torch.LongTensor(self.mem_size).fill_(0).to(self.device) if 'Sub' in args.agent else None
        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('buffer_input', buffer_input)
        self.register_buffer('buffer_label', buffer_label)
        self.register_buffer('buffer_logits', buffer_logits)
        self.register_buffer('buffer_sub', buffer_sub)

        # define update and retrieve method
        self.update_method = name_match.update_methods[args.update](self.params)
        self.retrieve_method = name_match.retrieve_methods[args.retrieve](self.params)

        if self.params.buffer_tracker:
            self.buffer_tracker = BufferClassTracker(n_classes[args.data], self.device)

    def update(self, x, y,**kwargs):
        return self.update_method.update(buffer=self, x=x, y=y, **kwargs)


    def retrieve(self, **kwargs):
        return self.retrieve_method.retrieve(buffer=self, **kwargs)
