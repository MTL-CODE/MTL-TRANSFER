import torch
from torch.nn.functional import softmax

def sequence_mask(x, valid_lens, value):
    """return mask according to lengths
    Args:
        x (tensor):
            [batch_size, len]
        valid_lens (tensor):
            [batch_size]
        value (float):
    """
    shape = x.shape
    device = x.device
    
    mask = (torch.arange(0, shape[1])
            .unsqueeze(dim=1)
            .repeat(1, shape[0])
            .lt(valid_lens) # broadcast
            .permute(1, 0)
            .to(device))
    return x.masked_fill(~mask, value)


def masked_softmax(x, valid_lens):
    return softmax(sequence_mask(x, valid_lens, -float('inf')), dim=-1)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def net_parameters(net):
    return [p for p in net.parameters() if p.requires_grad]


def data2tensor(batch, device):
    src = [torch.tensor(example[0], device=device) for example in batch]
    tgt = [torch.tensor(example[1], device=device) for example in batch]
    return src, tgt


def truncate_pad(line, length, padding_value):
    originLen = len(line)
    if(originLen > length):
        return line[:length], originLen
    return line + [padding_value] * (length - originLen), originLen


def clip_gradients(model, grad_clip_val):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm