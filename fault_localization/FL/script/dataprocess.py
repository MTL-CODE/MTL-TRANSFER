def add_bos_eos(opt, data):
    bos = opt['vocab']['tgt_bos']
    eos = opt['vocab']['tgt_eos']
    workType = ['train', 'valid', 'test']
    for type0 in workType:
        data[type0]['target'] = [[bos] + line + [eos] for line in data[type0]['target']]


def truncate_pad(data, num_steps, padding_token):
    if len(data) > num_steps:
        return data[:num_steps]  # 
    return data + [padding_token] * (num_steps - len(data))  # 


def align_data(opt, data):
    padding_idx = {'source': opt['vocab']['src_pad'], 'target': opt['vocab']['tgt_pad']}
    workType = ['train', 'valid', 'test']
    dataType = ['source', 'target']
    length = {}
    
    for type0 in workType:
        length[type0] = {}
        for type1 in dataType:
            length[type0][type1] = []
            for idx, line in enumerate(data[type0][type1]):
                length[type0][type1].append(len(line))
            print(type0 + " " + type1 + " max_len=" + str(max(length[type0][type1])))
    
    for type0 in workType:
        for type1 in dataType:
            ll = max(length[type0][type1])
            data[type0][type1] = [truncate_pad(
                line, ll, padding_idx[type1]) for line in data[type0][type1]]