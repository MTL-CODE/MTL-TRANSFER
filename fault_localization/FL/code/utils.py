import torch
from torch.nn.functional import softmax
import os
import pickle


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
            .lt(valid_lens)  # broadcast
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
    if (originLen > length):
        return line[:length], originLen
    return line + [padding_value] * (length - originLen), originLen


def clip_gradients(model, grad_clip_val):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm


def save_model(model, filePath):
    torch.save(model.state_dict(), os.path.join(filePath, "model_params.pkl"))
    torch.save(model, os.path.join(filePath, "model.pkl"))


def convert_outs(outs, dict):
    outs = torch.argmax(outs, dim=-1)
    idx_seq = outs.tolist()

    def convert(idxs, dict):
        if not isinstance(idxs, (list, tuple)):
            return dict.idx2token[idxs]
        return [convert(idx, dict) for idx in idxs]

    token_seq = convert(idx_seq, dict)

    return token_seq


def save_model(model, path):
    torch.save(model.state_dict(), os.path.join(path, "model_params.pkl"))
    torch.save(model, os.path.join(path, "model.pkl"))


def load_model(model, path):
    model.load_state_dict(torch.load(os.path.join(path, "model_params.pkl")))


def save_object(file, path):
    with open(path, "wb") as f:
        pickle.dump(file, path)


def load_object(file, path):
    with open(path, "rb") as f:
        return pickle.load(f)
