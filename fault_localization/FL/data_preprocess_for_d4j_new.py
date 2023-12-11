import os
import javalang
import pickle
from gensim.models.word2vec import Word2Vec
import numpy as np
import re
import string
import random
import sys
from MTL_model.data_util.data import Vocab
from MTL_model.data_util import data, config_all




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


def cut_data(token_seq, token_length_for_reserve): 
    if len(token_seq) <= token_length_for_reserve:
        return token_seq
    else:
        start_index = token_seq.index("rank2fixstart")
        end_index = token_seq.index("rank2fixend")
        assert end_index > start_index
        length_of_annotated_statement = end_index - start_index + 1
        if length_of_annotated_statement <= token_length_for_reserve:
            padding_length = token_length_for_reserve - length_of_annotated_statement
            # give 2/3 padding space to content before annotated statement
            pre_padding_length = padding_length // 3 * 2
            # give 1/3 padding space to content after annotated statement
            post_padding_length = padding_length - pre_padding_length
            if start_index >= pre_padding_length and len(token_seq) - end_index - 1 >= post_padding_length:
                return token_seq[start_index - pre_padding_length: end_index + 1 + post_padding_length]
            elif start_index < pre_padding_length:
                return token_seq[:token_length_for_reserve]
            elif len(token_seq) - end_index - 1 < post_padding_length:
                return token_seq[len(token_seq) - token_length_for_reserve:]
        else:
            return token_seq[start_index: start_index + token_length_for_reserve]




if __name__ == "__main__":

    pattern_list = ['InsertMissedStmt', 'InsertNullPointerChecker', 'MoveStmt', 'MutateConditionalExpr',
                    'MutateDataType', 'MutateLiteralExpr', 'MutateMethodInvExpr', 'MutateOperators',
                    'MutateReturnStmt', 'MutateVariable', 'RemoveBuggyStmt']


    
    for type_1 in pattern_list:
        print(type_1)
     
        current_pattern = type_1

        if current_pattern not in pattern_list:
            print("The fix pattern specified by the argument dost not exist.")
            exit(-1)

        print("Current fix pattern: {}".format(current_pattern))

     
        d4j_data_dir = "./d4j_data"

        new_data_dir = "./data/{}/".format(current_pattern)

        if not os.path.exists(new_data_dir):  
            os.makedirs(new_data_dir)

        print("Parameters declaration")
        token_length_for_reserve = 400

        vocab_path = "../../data/data-new/code_data_demo_{}/finished_files/vocab".format(current_pattern)
        src_vocab = Vocab(vocab_path, config_all.vocab_size)


        print("Data preprocessing for defects4j data")
        index_oov = src_vocab.word2id('[UNK]')
        index_padding = src_vocab.word2id('[PAD]')

        print("ovv pad index is :",index_oov,index_padding)

        with open(os.path.join(d4j_data_dir, "src_code.pkl"), "rb") as file:
            src_code = pickle.load(file)



        d4j_w2v_ = {}
        for project in src_code:
            samples = []
            for method in src_code[project]:
                # print("d4j origin code is:\n",method)
                method = method.strip()
                tokens = javalang.tokenizer.tokenize(method)

                token_seq = []

                for token in tokens:

                    if isinstance(token, javalang.tokenizer.String):
                        tmp_token = ["stringliteral"]
                    else:
                        tmp_token = solve_camel_and_underline(token.value)
                    token_seq += tmp_token
                token_seq = cut_data(token_seq, token_length_for_reserve)

               
                normal_record = [src_vocab.word2id(token) for token in token_seq] 
                
                print("the id data -- d4j: \n", normal_record)

                
                samples.append(normal_record)
            d4j_w2v_[project] = samples
            
        with open(os.path.join(new_data_dir, "d4j_w2v_new2.pkl"), "wb") as file:
            pickle.dump(d4j_w2v_, file)


        print(type_1 + "Done!!")
