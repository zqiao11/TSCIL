import torch


class Reservoir_update(object):
    def __init__(self, params):
        super().__init__()

    def update(self, buffer, x, y, logits=None, subjects=None, **kwargs):
        batch_size = x.size(0)

        # add whatever still fits in the buffer
        place_left = max(0, buffer.buffer_input.size(0) - buffer.current_index)
        if place_left:
            offset = min(place_left, batch_size)
            buffer.buffer_input[buffer.current_index: buffer.current_index + offset].data.copy_(x[:offset])
            buffer.buffer_label[buffer.current_index: buffer.current_index + offset].data.copy_(y[:offset])

            if logits is not None:
                buffer.buffer_logits[buffer.current_index: buffer.current_index + offset].data.copy_(logits[:offset])

            if subjects is not None:
                buffer.buffer_sub[buffer.current_index: buffer.current_index + offset].data.copy_(subjects[:offset])

            buffer.current_index += offset
            buffer.n_seen_so_far += offset

            # everything was added
            if offset == x.size(0):
                filled_idx = list(range(buffer.current_index - offset, buffer.current_index, ))
                if buffer.params.buffer_tracker:
                    buffer.buffer_tracker.update_cache(buffer.buffer_label, y[:offset], filled_idx)
                return filled_idx

        # remove what is already in the buffer
        x, y = x[place_left:], y[place_left:]

        if logits is not None:
            logits = logits[place_left:]

        if subjects is not None:
            subjects = subjects[place_left:]
        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, buffer.n_seen_so_far).long()
        valid_indices = (indices < buffer.buffer_input.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer   = indices[idx_new_data]

        buffer.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return []

        assert idx_buffer.max() < buffer.buffer_input.size(0)
        assert idx_buffer.max() < buffer.buffer_label.size(0)
        # assert idx_buffer.max() < self.buffer_task.size(0)

        assert idx_new_data.max() < x.size(0)
        assert idx_new_data.max() < y.size(0)

        idx_map = {idx_buffer[i].item(): idx_new_data[i].item() for i in range(idx_buffer.size(0))}

        replace_y = y[list(idx_map.values())]
        if buffer.params.buffer_tracker:
            buffer.buffer_tracker.update_cache(buffer.buffer_label, replace_y, list(idx_map.keys()))
        # perform overwrite op
        buffer.buffer_input[list(idx_map.keys())] = x[list(idx_map.values())]
        buffer.buffer_label[list(idx_map.keys())] = replace_y

        if logits is not None:
            buffer.buffer_logits[list(idx_map.keys())] = logits[list(idx_map.values())]

        if subjects is not None:
            buffer.buffer_sub[list(idx_map.keys())] = subjects[list(idx_map.values())]

        return list(idx_map.keys())