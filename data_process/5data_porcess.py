# /usr/bin/python
# -*- coding: utf-8 -*-
import io
import os
import sys
import time
import pickle

import struct
import collections
from tensorflow.core.example import example_pb2

import sys

fixed_type = sys.argv[1]


fixed_type_name = "code_data_demo_" + fixed_type
DATA_ROOT = "../data/data-new/" + fixed_type_name

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

VOCAB_SIZE = 50000
CHUNK_SIZE = 1000

FINISHED_FILE_DIR = os.path.join(DATA_ROOT, "finished_files")


def read_pkl(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def build_train_val(befor_data, after_data,label_data, train_num=600):
    train_list = []
    val_list = []
    n = 0
    i = 0
    j = 0
    for befor, after, label in zip(befor_data, after_data, label_data):
        n += 1
        if n <= train_num:
            train_list.append(befor)
            train_list.append(after)
            train_list.append(label)
            i += 1
        else:
            val_list.append(befor)
            val_list.append(after)
            val_list.append(label)
            j += 1
    print("data total is: {}, train data total is：{}, val data total is:{}".format(n, i, j))
    return train_list, val_list


def save_file(filename, li):
    with io.open(filename, 'w+', encoding='utf-8') as f:
        for item in li:
            f.write(item + u'\n')
    print("Save {} ok.".format(filename))


def chunk_file(finished_files_dir, chunks_dir, name, chunk_size):
    in_file = os.path.join(finished_files_dir, '%s.bin' % name)
    print(in_file)
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (name, chunk))  
        with open(chunk_fname, 'wb') as writer:
            for _ in range(chunk_size):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all(chunks_dir):
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    for name in ['train', 'val']:
        print("Splitting %s data into chunks..." % name)
        chunk_file(FINISHED_FILE_DIR, chunks_dir, name, CHUNK_SIZE)
    print("Saved chunked data in %s" % chunks_dir)


def read_text_file(text_file):
    lines = []
    with io.open(text_file, "r", encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def write_to_bin(input_file, out_file, makevocab=False):  
    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        lines = read_text_file(input_file)
        len_lines = len(lines)
        for i, _ in enumerate(lines):
            if i % 3 == 0:
                article = lines[i]
            if i % 3 == 1:
                abstract = "%s %s %s" % (SENTENCE_START, lines[i], SENTENCE_END)

                tf_example = example_pb2.Example()
                tf_example.features.feature['beforefix'].bytes_list.value.extend([bytes(article)])
                tf_example.features.feature['afterfix'].bytes_list.value.extend([bytes(abstract)])

                if( i+1 <= len_lines-1 ):
                    i = i + 1
                    label = lines[i]
                    tf_example.features.feature['label'].bytes_list.value.extend([bytes(label)])


                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, tf_example_str))

                if makevocab:
                    art_tokens = article.split(' ')
                    abs_tokens = abstract.split(' ')
                    abs_tokens = [t for t in abs_tokens if
                                  t not in [SENTENCE_START, SENTENCE_END]]  
                    tokens = art_tokens + abs_tokens
                    tokens = [t.strip() for t in tokens]  
                    tokens = [t for t in tokens if t != ""]  
                    vocab_counter.update(tokens)
                

    print("Finished writing file %s\n" % out_file)


    if makevocab:
        print("Writing vocab file...")
        with io.open(os.path.join(FINISHED_FILE_DIR, "vocab"), 'w', encoding='utf-8') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


if __name__ == '__main__':

    train_data_path = '../data/data-diff/'+fixed_type+'/train_python2.pkl'
    valid_data_path = '../data/data-diff/'+fixed_type+'/valid__python2.pkl'
    test_data_path = '../data/data-diff/'+fixed_type+'/test__python2.pkl'

    train_data = read_pkl(train_data_path)
    valid_data = read_pkl(valid_data_path)
    test_data = read_pkl(test_data_path)

    beforfix_data_list = []

    for i in range(len(train_data['source'])):
        word_list = (train_data['source'][i])
        beforfix_data_list.append(' '.join(word_list).strip())

    for i in range(len(valid_data['source'])):
        word_list = (valid_data['source'][i])
        beforfix_data_list.append(' '.join(word_list).strip())

    for i in range(len(test_data['source'])):
        word_list = (test_data['source'][i])
        beforfix_data_list.append(' '.join(word_list).strip())

    print("read beforfix data successful ： data len and No.1000 data is",len(beforfix_data_list), (beforfix_data_list[10]))

    afterfix_data_list = []

    for i in range(len(train_data['target'])):
        word_list = (train_data['target'][i])
        afterfix_data_list.append(' '.join(word_list).strip())

    for i in range(len(valid_data['target'])):
        word_list = (valid_data['target'][i])
        afterfix_data_list.append(' '.join(word_list).strip())

    for i in range(len(test_data['target'])):
        word_list = (test_data['target'][i])
        afterfix_data_list.append(' '.join(word_list).strip())

    print("read afterfix data successful ： data len and No.1000 data is",len(afterfix_data_list), (afterfix_data_list[10]))


    label_data_list = []
    for i in range(len(train_data['label'])):
        label_data_list.append(str(train_data['label'][i]))

    for i in range(len(valid_data['label'])):
        label_data_list.append(str(valid_data['label'][i]))

    for i in range(len(test_data['label'])):
        label_data_list.append(str(test_data['label'][i]))

    print("read label data successful ： data len and No.1000 data is", len(label_data_list), label_data_list[10])


    if not os.path.isdir(DATA_ROOT):
        os.mkdir(DATA_ROOT)


    beforfix_data = beforfix_data_list
    afterfix_data = afterfix_data_list
    label_data = label_data_list


    train_split = len(afterfix_data_list)*0.8  
    train_list, val_list = build_train_val(beforfix_data, afterfix_data, label_data,  train_num=train_split)




    train_file = os.path.join(DATA_ROOT, "train_code_art_sum_prep.txt")
    val_file = os.path.join(DATA_ROOT, "val_code_art_sum_prep.txt")

    save_file(train_file, train_list)
    save_file(val_file, val_list)

    if not os.path.exists(FINISHED_FILE_DIR):
        os.makedirs(FINISHED_FILE_DIR)

    chunks_dir = os.path.join(FINISHED_FILE_DIR, 'chunked')

    write_to_bin(val_file, os.path.join(FINISHED_FILE_DIR, "val.bin"))
    write_to_bin(train_file, os.path.join(FINISHED_FILE_DIR, "train.bin"), makevocab=True)
    chunk_all(chunks_dir)


    d4j_dict_data_path = "../data/data_d4j/d4j_219_token_dict_python2.pkl"
    d4j_dict_data = read_pkl(d4j_dict_data_path)

    d4j_beforfix_data_list = []
    for key_name1 in d4j_dict_data.keys():
        # print("the dictionary data, key:{},value:{}".format(key_name1,d4j_dict_data[key_name1]))
        word_list = d4j_dict_data[key_name1]
        d4j_beforfix_data_list.append(' '.join(word_list).strip())
        d4j_beforfix_data_list.append(key_name1 + '==== ' +' '.join(word_list).strip())
        d4j_beforfix_data_list.append('1')

    for i in d4j_beforfix_data_list:
        print((i))

    print("read d4j beforfix data successful, the data len and No.10 data is:",len(d4j_beforfix_data_list),d4j_beforfix_data_list[10])

    d4j_test_file = os.path.join(DATA_ROOT, "d4j_code_art_prep.txt")
    save_file(d4j_test_file, d4j_beforfix_data_list)

    write_to_bin(d4j_test_file, os.path.join(FINISHED_FILE_DIR, "d4j_test.bin"))















