
import json
import os
import sys
import torch
import time
import numpy as np
import pickle
import random
import torch.nn as nn
import torch.nn.functional as F



from MTL_model.train.model import Model

from MTL_model.data_util import data, config_all
use_cuda = config_all.use_gpu and torch.cuda.is_available()



def load_from_file(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


if __name__ == "__main__":
    fix_patterns = ["InsertMissedStmt", "InsertNullPointerChecker", "MoveStmt", "MutateConditionalExpr",
                    "MutateDataType", "MutateLiteralExpr", "MutateMethodInvExpr", "MutateOperators",
                    "MutateReturnStmt", "MutateVariable", "RemoveBuggyStmt"]

    # fix_patterns = ["InsertMissedStmt"]
    out_semantic_features = {}
    for fix_pattern in fix_patterns:
        print("Fix pattern: {}".format(fix_pattern))
        root = "../data/{}/".format(fix_pattern)

        checker_info_path = "../d4j_data/checker_info.pkl"
        checker_info = load_from_file(checker_info_path)

        d4j_data = load_from_file(os.path.join(root, "d4j_w2v_new2.pkl")) 
       
        model_path = "../../data/data-new/code_data_demo_{}/Log/save_model/model/model_mlp_best".format(fix_pattern)
        multi_model = Model(model_path, is_eval=True)


        for project in d4j_data:
            q = 0
            input_samples = d4j_data[project]
     

            data_len = len(input_samples)  

            every_data = [ input_samples[i:i+40] for i in range(0,data_len,40) ]
            every_data_len = len(every_data) 

            out_puts = []
            for k in range(every_data_len):
                input_data = every_data[k]  
                new_input_data = []
                for every_data_ids in input_data:
                    if len(every_data_ids) < 400:
                        every_data_ids += [0] * (400 - len(every_data_ids))
                    new_input_data.append(every_data_ids)


                input_data = torch.tensor(new_input_data)

                input_data_lens = [len(sample) for sample in input_data ]
                input_data_lens = torch.tensor(input_data_lens)

                if use_cuda:
                    input_data = input_data.cuda()
                    
                encoder_outputs, encoder_feature, encoder_hidden = multi_model.encoder(input_data,input_data_lens)
                mlp_output = multi_model.mlp(encoder_outputs)
                
                output = torch.softmax(mlp_output, dim=-1) 
                output = output[:, 0].cpu().tolist()
                
                out_puts.extend(output)


            if project not in out_semantic_features:
                out_semantic_features[project] = []

            for index, checker_flag in enumerate(checker_info[project]):
                if fix_patterns.index(fix_pattern) == 0:
                    if checker_flag[fix_patterns.index(fix_pattern)] == 0:
                        out_semantic_features[project].append([0])
                    else:
                        out_semantic_features[project].append([out_puts[index]])
                else:
                    if checker_flag[fix_patterns.index(fix_pattern)] == 0:
                        out_semantic_features[project][index].append(0)
                    else:
                        out_semantic_features[project][index].append(out_puts[index])

        # f.close()

    with open("./d4j_data/semantic_new.pkl", "wb") as file:
        pickle.dump(out_semantic_features, file)


        









