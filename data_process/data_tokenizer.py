
import pickle
import javalang
import os
import re
import string
import json


def read_pkl(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def write_pkl(filepath, obj):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def read_file(filepath):
    with open(filepath, "r") as f:
        return f.readlines()


def write_file(filepath, str):
    with open(filepath, "w") as f:
        f.write(str)


def print_dict(obj, str):
    if not isinstance(obj, dict):
        return
    for key in obj.keys():
        print(str + key)
        if isinstance(obj[key], dict):
            print_dict(obj[key], str + ' ')


def solve_camel_and_underline(token):
    if token.isdigit():
        return [token]
    else:
        p = re.compile(r'([a-z]|\d)([A-Z])')
        sub = re.sub(p, r'\1_\2', token).lower()
        sub_tokens = sub.split("_")
        tokens = re.sub(" +", " ", " ".join(sub_tokens)).strip()
        final_token = []
        for factor in tokens.split(" "):
            final_token.append(factor.rstrip(string.digits))
        return final_token


def javalang_tokens_solve(javalang_tokens):
    first_tokens = []
    tokens = []
    for token in javalang_tokens:
        first_tokens.extend([token.value])

        if isinstance(token, javalang.tokenizer.String):
            tmp_token = ["stringliteral"]
        else:
            tmp_token = solve_camel_and_underline(token.value)
        tokens += tmp_token
        final_tokens = tokens


    return first_tokens, final_tokens


if __name__ == '__main__':

    dataset_pkl_path = '../data/dataset_pre_diff.pkl'
    # positive:
    # negative: 
    # positive_pre_diff

    dataset = read_pkl(dataset_pkl_path)
    faultTypes = dataset.keys()
    # faultTypes = ['MutateDataType']

    for faultType in faultTypes:
        data = []
        
        print(f"[{faultType}]")
        data_length = 0
        for type in dataset[faultType].keys():
            if(data_length != 0):
                assert data_length == len(dataset[faultType][type])
            else:
                data_length = len(dataset[faultType][type])
        print(f"length: {data_length}")
        beforefix_len = []
        afterfix_len  = []
        for i in range(data_length):    
            data_text_positive = {}
            data_text_negative = {}

            # data_text_positive['beforefix'] = javalang_tokens_solve(javalang.tokenizer.tokenize(dataset[faultType]['positive'][i]))
            # data_text_positive['afterfix']  = javalang_tokens_solve(javalang.tokenizer.tokenize(dataset[faultType]['positive_pre_diff'][i]))
            # data_text_positive['label']     = 1
            #
            # data_text_negative['beforefix'] = javalang_tokens_solve(javalang.tokenizer.tokenize(dataset[faultType]['negative'][i]))
            # data_text_negative['afterfix']  = ['TRANFERSEXTENDNOCHANGE']
            # data_text_negative['label']     = 0;

            pos_first_token_result, pos_final_token_result = javalang_tokens_solve(javalang.tokenizer.tokenize(dataset[faultType]['positive'][i]))
            diff_first_token_result, diff_final_token_result = javalang_tokens_solve( javalang.tokenizer.tokenize(dataset[faultType]['positive_pre_diff'][i]) )
            neg_first_first_token_result, neg_final_token_result = javalang_tokens_solve(javalang.tokenizer.tokenize(dataset[faultType]['negative'][i]))

            data_text_positive['beforefix'] = pos_first_token_result
            data_text_positive['afterfix'] = diff_first_token_result
            data_text_positive['label'] = 1

            data_text_negative['beforefix'] = neg_first_first_token_result
            data_text_negative['afterfix'] = ['TRANFERSEXTENDNOCHANGE']
            data_text_negative['label'] = 0;

            print ('1.pos beforefix code token: ', data_text_positive['beforefix'])
            print ('2.pos afterfix code token: ', data_text_positive['afterfix'])
            print ('3.pos label: ',data_text_positive['label'])

            print ('1.neg beforefix code token: ', data_text_negative['beforefix'])
            print ('2.neg afterfix code token: ', data_text_negative['afterfix'])
            print ('3.neg label: ', data_text_negative['label'])


            data.append(data_text_positive.copy())
            data.append(data_text_negative.copy())

            beforefix_len.append(str(len(data_text_positive['beforefix'])))
            beforefix_len.append(str(len(data_text_negative['beforefix'])))
            afterfix_len.append(str(len(data_text_positive['afterfix'])))
            afterfix_len.append(str(len(data_text_negative['afterfix'])))
        data_text_length = len(data)

        write_path = os.path.join("../data/data-diff", faultType)
        os.makedirs(write_path, exist_ok=True)
        write_pkl(os.path.join(write_path, 'data_token.pkl'), data)

        open( os.path.join(write_path,'data_token.txt') , 'w').write(json.dumps(data))  

        write_file(os.path.join(write_path, 'beforefix_length.txt'), '\n'.join(beforefix_len))
        write_file(os.path.join(write_path, 'afterfix_length.txt'), '\n'.join(afterfix_len))
