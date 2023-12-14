
import pickle
import javalang
import os
import re

"""      
   Input: ../data/data_prepare/dataset.pkl
   Output: "../data/data_prepare/dataset_raw_diff.pkl" 、"../data/data_prepare/dataset_pre_diff.pkl"
"""


def read_pkl(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def write_pkl(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


print("### Reading dataset_pre.pkl")
dataset_pre_pkl_path = '../data/dataset.pkl'
dataset_pre_pkl = read_pkl(dataset_pre_pkl_path)

faultType = dataset_pre_pkl.keys()
codeType = ['positive', 'negative', 'patch']

for type1 in faultType:
    for type1_1 in codeType:
        print(type1 + " " + type1_1)
        list1 = dataset_pre_pkl[type1][type1_1]
        file_path = '../data/data_prepare/' + type1
        if not os.path.isdir(file_path):
            os.makedirs(file_path)
        write_pkl(list1, file_path + '/' + type1_1 + '.pkl')
        # print( read_pkl(file_path + '/'+ type1_1 + '.pkl') )

        file_path1_1 = file_path + '/' + type1_1
        if not os.path.isdir(file_path1_1):
            os.makedirs(file_path1_1)

for type1 in faultType:
    for type1_1 in codeType:
        txt_path = '../data/data_prepare/' + type1 + '/' + type1_1
        data_path = '../data/data_prepare/' + type1 + '/' + type1_1 + '.pkl'
        data = read_pkl(data_path)

        for i in range(len(data)):
            list11 = data[i]
            num_context_path = txt_path + '/' + type1_1 + '_' + str(i) + '.txt'
            open(num_context_path, 'w').write(list11)

for type1 in faultType:

    positive_data_path = '../data/data_prepare/' + type1 + '/positive.pkl'
    positive_data = read_pkl(positive_data_path)

    pos_raw_diff_txt_path = '../data/data_prepare/' + type1 + '/positive_raw_diff.txt'
    file_pos_raw_diff_txt = open(pos_raw_diff_txt_path, 'w')

    pos_pre_diff_txt_path = '../data/data_prepare/' + type1 + '/positive_pre_diff.txt'
    file_pos_pre_diff_txt = open(pos_pre_diff_txt_path, 'w')

    positive_raw_diff_list = []  
    positive_pre_diff_list = []  
    for i in range(len(positive_data)):

        txt_path_postive = '../data/data_prepare/' + type1 + '/' + 'positive' + '/'
        txt_path_postive_patch = '../data/data_prepare/' + type1 + '/' + 'patch' + '/'

        txt_file_postive = txt_path_postive + 'positive_' + str(i) + '.txt'
        txt_file_postive_patch = txt_path_postive_patch + 'patch_' + str(i) + '.txt'

        result_pos_raw_diff = os.popen("diff -w " + txt_file_postive + ' ' + txt_file_postive_patch).read()
        print(type(result_pos_raw_diff), i)

        positive_raw_diff_list.append(result_pos_raw_diff)
        file_pos_raw_diff_txt.write(result_pos_raw_diff)

        result_pos_pre_diff = result_pos_raw_diff

        pattern_c1 = re.compile('(\d*)(c)(\d+\n<)', re.S)
        pattern_c2 = re.compile('(\d*)(c)(\d+)(,)(\d+\n<)', re.S)
        pattern_c3 = re.compile('(\d*),(\d*)(c)(\d+\n<)', re.S)
        pattern_c4 = re.compile('(\d*),(\d*)(c)(\d+),(\d+\n<)', re.S)

        pattern_a1 = re.compile('(\d*)(a)(\d+\n>)', re.S)
        pattern_a2 = re.compile('(\d*)(a)(\d+),(\d+\n>)', re.S)

        pattern_d1 = re.compile('(\d*)(d)(\d+\n<)', re.S)
        pattern_d2 = re.compile('(\d*),(\d*)(d)(\d+\n<)', re.S)

        result_pos_pre_diff = re.sub(pattern_c4, " ", result_pos_pre_diff)
        result_pos_pre_diff = re.sub(pattern_c3, " ", result_pos_pre_diff)
        result_pos_pre_diff = re.sub(pattern_c2, " ", result_pos_pre_diff)
        result_pos_pre_diff = re.sub(pattern_c1, " ", result_pos_pre_diff)

        result_pos_pre_diff = re.sub(pattern_a2, "TRANFEREXTENDINSERT", result_pos_pre_diff)
        result_pos_pre_diff = re.sub(pattern_a1, "TRANFEREXTENDINSERT", result_pos_pre_diff)

        result_pos_pre_diff = re.sub(pattern_d2, "TRANSFEREXTENDDEL", result_pos_pre_diff)
        result_pos_pre_diff = re.sub(pattern_d1, "TRANSFEREXTENDDEL", result_pos_pre_diff)

        result_pos_pre_diff = result_pos_pre_diff.replace("\n---\n", "CHANGE--TRANSFEREXTENDCHANGE")
        result_pos_pre_diff = result_pos_pre_diff.replace(">", "")
        result_pos_pre_diff = result_pos_pre_diff.replace("<", "")
        result_pos_pre_diff = result_pos_pre_diff.replace("rank2fixstart", "")
        result_pos_pre_diff = result_pos_pre_diff.replace("rank2fixend", "")

        if( "CHANGE--TRANSFEREXTENDCHANGE" in  result_pos_pre_diff ):
            result_pos_pre_diff = result_pos_pre_diff.split('CHANGE--')[1]

        print (type1,"id:{} diff:".format(i),result_pos_pre_diff)


        positive_pre_diff_list.append(result_pos_pre_diff)
        file_pos_pre_diff_txt.write(result_pos_pre_diff)

    write_pkl(positive_raw_diff_list, '../data/data_prepare/' + type1 + '/positive_raw_diff.pkl')
    write_pkl(positive_pre_diff_list, '../data/data_prepare/' + type1 + '/positive_pre_diff.pkl')

    file_pos_raw_diff_txt.close()
    file_pos_pre_diff_txt.close()

    print("### Reading positive_diff.pkl")
    pos_raw_diff_pkl_path = '../data/data_prepare/' + type1 + '/positive_raw_diff.pkl'
    pos_raw_diff_pkl = read_pkl(pos_raw_diff_pkl_path)
    print(pos_raw_diff_pkl[0:50])

    pos_pre_diff_pkl_path = '../data/data_prepare/' + type1 + '/positive_pre_diff.pkl'
    pos_pre_diff_pkl = read_pkl(pos_pre_diff_pkl_path)
    print(pos_pre_diff_pkl[0:50])


print("### Reading positive.pkl、negative.pkl、positive_raw_diff.pkl  or  positive.pkl、negative.pkl、positive_pre_diff.pkl")

dict_raw_data = {}
dict_pre_data = {}

for type1 in faultType:

    data_positive_path = '../data/data_prepare/' + type1 + '/positive.pkl'
    data_positive = read_pkl(data_positive_path)
    data_positive_len = len(data_positive)

    data_negative_path = '../data/data_prepare/' + type1 + '/negative.pkl'
    data_negative = read_pkl(data_negative_path)
    data_negative_len = len(data_negative)

    data_pos_raw_diff_path = '../data/data_prepare/' + type1 + '/positive_raw_diff.pkl'
    data_pos_raw_diff = read_pkl(data_pos_raw_diff_path)
    data_pos_raw_diff_len = len(data_pos_raw_diff)

    data_pos_pre_diff_path = '../data/data_prepare/' + type1 + '/positive_pre_diff.pkl'
    data_pos_pre_diff = read_pkl(data_pos_pre_diff_path)
    data_pos_pre_diff_len = len(data_pos_pre_diff)

    print(type1, "data len:", data_positive_len, data_negative_len, data_pos_raw_diff_len, data_pos_pre_diff_len)

    code_list_pos = []
    code_list_neg = []
    code_list_pos_raw_diff = []
    code_list_pos_pre_diff = []

    for i in range(data_positive_len):
        data_pos = data_positive[i]

        data_pos_raw_diff_patch = data_pos_raw_diff[i]
        data_pos_raw_diff_patch = data_pos_raw_diff_patch.replace("\ No newline at end of file", " ")

        data_pos_pre_diff_patch = data_pos_pre_diff[i]
        data_pos_pre_diff_patch = data_pos_pre_diff_patch.replace("\ No newline at end of file", " ")

        data_neg = data_negative[i]

        code_list_pos.append(data_pos)  
        code_list_neg.append(data_neg) 
        code_list_pos_raw_diff.append(data_pos_raw_diff_patch)  
        code_list_pos_pre_diff.append(data_pos_pre_diff_patch)  

    print(type1, "code data len:", len(code_list_pos),
          len(code_list_neg), len(code_list_pos_raw_diff), len(code_list_pos_pre_diff))

    dict_raw_data[type1] = {'positive': code_list_pos, 'negative': code_list_neg,
                            'positive_raw_diff': code_list_pos_raw_diff}
    dict_pre_data[type1] = {'positive': code_list_pos, 'negative': code_list_neg,
                            'positive_pre_diff': code_list_pos_pre_diff}

print(dict_raw_data.keys())
write_pkl(dict_raw_data, "../data/data_prepare/dataset_raw_diff.pkl")

print(dict_pre_data.keys())
write_pkl(dict_pre_data, "../data/data_prepare/dataset_pre_diff.pkl")
