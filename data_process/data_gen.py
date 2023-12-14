import pickle
import javalang
import os
import re
import string
import random


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


def work(faultType):
    print(f"[{faultType}]")
    
    faultTypePath = os.path.join('../data/data-diff', faultType)
    dataPath = os.path.join(faultTypePath, 'data_token.pkl')
    dataToken = read_pkl(dataPath)
    
    maxBeforeFixLength = 500
    maxAfterFixLength = 200
    print(f"maxBeforeFixLength: {maxBeforeFixLength}")
    print(f"maxAfterFixLength : {maxAfterFixLength}")
    
    print(f"Before: {len(dataToken)}")
    tmp = []
    for i in range(0, len(dataToken), 2):
        if (len(dataToken[i]['beforefix']) <= maxBeforeFixLength) and (len(dataToken[i]['afterfix']) <= maxAfterFixLength) and (len(dataToken[i + 1]['beforefix']) <= maxBeforeFixLength) and (len(dataToken[i + 1]['afterfix']) <= maxAfterFixLength):
            tmp.append(dataToken[i])
            tmp.append(dataToken[i + 1])
    dataToken = tmp
    
    print(f"After : {len(dataToken)}")
    
    data_num = len(dataToken)
    train_num = int(0.8 * data_num)
    valid_num = int(0.1 * data_num)
    shuffleSeed = 7890
    
    idx = list(range(data_num))
    random.seed(shuffleSeed)
    random.shuffle(idx)
    
    train = {'source': [], 'target': [], 'label': []}
    valid = {'source': [], 'target': [], 'label': []}
    test = {'source': [], 'target': [], 'label': []}
    
    for i, id in enumerate(idx):
        if i < train_num:
            train['source'].append(dataToken[id]['beforefix'])
            train['target'].append(dataToken[id]['afterfix'])
            train['label'].append(dataToken[id]['label'])
        elif i < train_num + valid_num:
            valid['source'].append(dataToken[id]['beforefix'])
            valid['target'].append(dataToken[id]['afterfix'])
            valid['label'].append(dataToken[id]['label'])
        else:
            test['source'].append(dataToken[id]['beforefix'])
            test['target'].append(dataToken[id]['afterfix'])
            test['label'].append(dataToken[id]['label'])
            
    print(f"train num: {len(train['label'])}")
    print(f"valid num: {len(valid['label'])}")
    print(f"test  num: {len(test['label'])}")
    
    writePath = faultTypePath
    write_pkl(os.path.join(writePath, "train.pkl"), train)
    write_pkl(os.path.join(writePath, "valid.pkl"), valid)
    write_pkl(os.path.join(writePath, "test.pkl"), test)

if __name__ == '__main__':
    # work('MutateDataType')
    
    faultTypes = []
    data_path = '../data/data-diff'
    for root, dirs, files in os.walk(data_path):
        if 'data_token.pkl' in files:
            faultTypes.append(os.path.split(root)[-1])
    print (faultTypes)

    for faultType in faultTypes:
        work(faultType)
