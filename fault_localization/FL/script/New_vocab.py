import collections

def count_tokens(tokens):
    if isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    def __init__(self, tokens=None, max_size=30000, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        self.counter = count_tokens(tokens)
        self._token_freqs = sorted(self.counter.items(), key=lambda x: x[1], reverse=True)
        self.idx2token = reserved_tokens
        self.token2idx = {token: idx for idx, token in enumerate(self.idx2token)}
        for token, freq in self._token_freqs:
            if freq < min_freq or len(self.idx2token) == max_size:
                break
            if token not in self.token2idx:
                self.idx2token.append(token)
                self.token2idx[token] = len(self.idx2token) - 1

    def __len__(self):
        return len(self.idx2token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token2idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    @property 
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def build_vocab(src, tgt):
    min_freq = 5
    max_size = 30000
    share_vocab = True
    
    if share_vocab is True:
        src_vocab = tgt_vocab =  Vocab(src + tgt, max_size=max_size, min_freq=min_freq, reserved_tokens=['<unk>', '<pad>', '<bos>', '<eos>'])
    else:
        src_vocab = Vocab(src, max_size=max_size, min_freq=min_freq, reserved_tokens=['<unk>', '<pad>'])
        tgt_vocab = Vocab(tgt, max_size=max_size, min_freq=min_freq, reserved_tokens=['<unk>', '<pad>', '<bos>', '<eos>'])
    

    return src_vocab, tgt_vocab


def data_convert(data, src_vocab, tgt_vocab): 
    data = src_vocab[data]