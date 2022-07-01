import os, random, helper, plotille, numpy as np, random, file_manipulation, csv
from tabnanny import filename_only
from pickle import TRUE
import termplotlib as tpl
import matplotlib.pyplot as plt
from time import sleep
from progress.bar import Bar
import pandas as pd
import re
import csv
import sys
import yaml
from yaml.loader import SafeLoader
from pathlib import Path

path_to_testdirectory = os.path.dirname(os.path.abspath(sys.argv[1])) 

def read_yaml_file():
    '''reads a csv file with the test parameters specifying:

        Fuzzy_hashing_algorithms: ...
        Path_to_testfiles: ...
        Test_description: ...
    
    returns a  [ list | dict ]
    
    '''
    config_path = sys.argv[1]

    # Open the file and load the file
    with open(config_path) as f:
        data = yaml.load(f, Loader=SafeLoader)
        # print(yaml.dump(data))
    
    return data

def create_evaluation_data_dict_from_testfiles(fuzzy_hashes, dir_path, dataset_label):
    '''takes every file in a dir_path and hashes it, it collects information about the file as well. 
       The result is a dict with lists with the syntax:

       
       {
        "filename":[filesize, fragment_size, ..., fuzzy_hash_a, fuzzy_hash_b],
        ...
        }
        
       :param fuzzy_hashes: list of fuzzy hashes
       :param dir_path: path to the dir that contains all the files
       :dataset_label: for the this dataset
    '''

    evaluation_data_dict = {}

    for file in os.listdir(dir_path):

        evaluation_data_dict[file] = {}

        file_path = os.path.join(dir_path, file)
        
        #add the file size to dict
        evaluation_data_dict[file]["filesize"] =  helper.getfilesize(file_path)

        #write the fragment_size to dict
        evaluation_data_dict[file]["fragment_size"] = "placeholder"

        #write the filepath to dict
        evaluation_data_dict[file]["filepath"] = file_path

        #label the file (either anomaly or normal)
        if file == "anomaly":
            evaluation_data_dict[file]["label"] = "anomaly"
        else:
            evaluation_data_dict[file]["label"] = dataset_label
        
        for fuzzy_hash in fuzzy_hashes:
            fuzzy_hash_instance = helper.get_algorithm(fuzzy_hash)

            # fuzzy_hash of a file | three hashes do not return hashes as string so write the path to the file instead
            if fuzzy_hash not in ["FBHASH", "MRSHCF", "MRSHV2"]:
                hash_exp = fuzzy_hash_instance.get_hash(file_path)
                evaluation_data_dict[file][fuzzy_hash] = hash_exp
            else:
                evaluation_data_dict[file][fuzzy_hash] = False


    return evaluation_data_dict

def approximate_matching_anomaly_vs_files(evaluation_data_dict: dict):
    ''' takes the "anomaly" file in the dict and compares it to all other files

    '''

    # in the dict all keys after "fragment_size" are hashes ... lets collect them
    available_fuzzy_hashes = []
    keys =  evaluation_data_dict.get('anomaly', {}).keys()
    for elem in keys:
        if elem.isupper():
            available_fuzzy_hashes.append(elem)

    #TODO: needs to be calculated per fuzzy hash 

    #overall TP, FP, etc., counters for a single dataset
    tp_ctr = 0
    fp_ctr = 0
    tn_ctr = 0 
    fn_ctr = 0

    # total size of all tp or fp etc. files
    total_tp_size = 0
    total_fp_size = 0
    total_tn_size = 0
    total_fn_size = 0

    # total simscores of all tp or fp etc. files
    total_tp_score = 0
    total_fp_score = 0
    total_tn_score = 0
    total_fn_score = 0

    ctr = 0

    for testfile in evaluation_data_dict: #.items():

        # no sense in comparing anomaly with itself which is why this case is skipped
        if testfile != "anomaly":
            for fuzzy_hash in available_fuzzy_hashes:

                #create an instance of the fuzzy hash 
                fuzzy_hash_instance = helper.get_algorithm(fuzzy_hash)

                # in case that the fuzzy hash does not return a hash as a string (False is set in the dict)
                # the two testfiles are directly compared via their filepaths
                if evaluation_data_dict[testfile][fuzzy_hash] == False:

                    anomaly_filepath = evaluation_data_dict["anomaly"]["filepath"]
                    testfile_path = evaluation_data_dict[testfile]["filepath"]

                    similarity_score = fuzzy_hash_instance.compare_file_against_file(anomaly_filepath, testfile_path)

                # ... in case the fuzzy hash is already in the dict as a string ...
                else:

                    anomaly_hash = evaluation_data_dict["anomaly"][fuzzy_hash]
                    file_hash = evaluation_data_dict[testfile][fuzzy_hash]
                    similarity_score = fuzzy_hash_instance.compare_hash(anomaly_hash, file_hash)
                
                evaluation_data_dict[testfile]["sim_score {} anomaly_vs_testfile".format(fuzzy_hash)] = similarity_score

                #-TP : the anomaly was correctly discovered 
                #-FP : the anomaly was incorrectly discovered in a file without any anomaly
                #-FN : the anomaly was not discovered in a file with anomaly
                #-TN : a file without anomaly was correctly determined as beeing normal (similarity score 0)
                filesize = evaluation_data_dict[testfile]["filesize"]

                file_label = evaluation_data_dict[testfile]["label"]

                # when the similarity score is > 0 and the file is labeled as an anomaly (TP)
                if similarity_score > 0 and file_label == "anomaly":
                    tp_ctr += 1
                    evaluation_data_dict[testfile]["p&r {} anomaly_vs_testfile".format(fuzzy_hash)] = "tp"
                    total_tp_size += filesize
                    total_tp_score += similarity_score

                # when the similarity score is = 0 and the file is labeled as an anomaly (FN)             
                elif similarity_score == 0 and file_label == "anomaly":
                    fn_ctr += 1
                    evaluation_data_dict[testfile]["p&r {} anomaly_vs_testfile".format(fuzzy_hash)] = "fn"
                    total_fn_size += filesize
                    total_fn_score += similarity_score

                # when the similarity score is > 0 and the file is labeled as normal (FP)
                elif similarity_score > 0 and file_label == "normal":
                    fp_ctr += 1
                    evaluation_data_dict[testfile]["p&r {} anomaly_vs_testfile".format(fuzzy_hash)] = "fp"
                    total_fp_size += filesize
                    total_fp_score += similarity_score

                # when the similarity score is = 0 and the file is labeled as normal (TN)
                elif similarity_score == 0 and  file_label == "normal":
                    tn_ctr += 1
                    evaluation_data_dict[testfile]["p&r {} anomaly_vs_testfile".format(fuzzy_hash)] = "tn"
                    total_tn_size += filesize
                    total_tn_score += similarity_score

                else: 
                    print("ERROR, Similarity Score ", fuzzy_hash, " ", 
                        similarity_score," label: " , evaluation_data_dict[testfile]["label"])


    #TODO: this needs to be calculated for every fuzzy hash individually.
    print("TP: ", tp_ctr, 
          " FP: ", fp_ctr, 
          " TN: ", tn_ctr, 
          " FN: ", fn_ctr) 

    avg_tp_size = total_tp_size/tp_ctr if tp_ctr else 0
    avg_fp_size = total_fp_size/fp_ctr if fp_ctr else 0
    avg_tn_size = total_tn_size/tn_ctr if tn_ctr else 0
    avg_fn_size = total_fn_size/fn_ctr if fn_ctr else 0  

    print("avg_tp_size: ", avg_tp_size, 
          " avg_fp_size: ", avg_fp_size, 
          " avg_tn_size: ", avg_tn_size, 
          " avg_fn_size: ", avg_fn_size)

    avg_tp_score = total_tp_score/tp_ctr if tp_ctr else 0
    avg_fp_score = total_fp_score/fp_ctr if fp_ctr else 0
    avg_tn_score = total_tn_score/tn_ctr if tn_ctr else 0
    avg_fn_score = total_fn_score/fn_ctr if fn_ctr else 0

    print("avg_tp_score: ", avg_tp_score, 
          " avg_fp_score: ", avg_fp_score, 
          " avg_tn_score: ", avg_tn_score, 
          " avg_fn_score: ", avg_fn_score)


    save_path = path_to_testdirectory + '/results.yml'

    if os.path.exists(save_path):
        write_mode = 'a' # append if already exists
    else:
        write_mode = 'w' # make a new file if not
        fle = Path(save_path)
        fle.touch(exist_ok=True)

    with open(save_path, write_mode) as outfile:
        yaml.dump(evaluation_data_dict, outfile, default_flow_style=False)

    return evaluation_data_dict

def precision_and_recall_calculation(testfile_data_dict):
    '''takes a results file as input and evaluates it based on the labels. Records are:

        -TP : the anomaly was correctly discovered 
        -FP : the anomaly was incorrectly discovered in a file without any anomaly
        -FN : the anomaly was not discovered in a file with anomaly
        -TN : a file without anomaly was correctly determined as beeing normal (similarity score 0)

        - Sizes of TP, FP, FN, TN files
        - distribution of file types
        - avg sim score of TP, FP, FN, TN files should the 
    '''
    
    # create a dict which holds all the evaluation result data
    #testfile_data_dict = read_yaml_file() 

    #overall TP, FP, etc., counters for a single dataset
    tp_ctr = 0
    fp_ctr = 0
    tn_ctr = 0 
    fn_ctr = 0
    

    for testfile in testfile_data_dict:

        #get all fuzzy hashes that a file has been hashed with
        available_fuzzy_hashes = []

        keys =  testfile_data_dict.get(testfile, {}).keys()
        for elem in keys:
            if elem.isupper():
                available_fuzzy_hashes.append(elem)

        for fuzzy_hash in available_fuzzy_hashes:

            
            sim_score = testfile_data_dict[testfile]["sim_score|{}|anomaly_vs_testfile".format(fuzzy_hash)]
            file_label = testfile_data_dict[testfile]["label"]

            # when the similarity score is > 0 and the file is labeled as an anomaly (TP)
            if sim_score > 0 and file_label == "anomaly":
                tp_ctr += 1
                testfile_data_dict[testfile]["p&r|{}|anomaly_vs_testfile".format(fuzzy_hash)] = "tp"

            # when the similarity score is = 0 and the file is labeled as an anomaly (FN)             
            elif sim_score == 0 and file_label == "anomaly":
                fn_ctr += 1
                testfile_data_dict[testfile]["p&r|{}|anomaly_vs_testfile".format(fuzzy_hash)] = "fn"

            # when the similarity score is > 0 and the file is labeled as normal (FP)
            elif sim_score > 0 and file_label == "normal":
                fp_ctr += 1
                testfile_data_dict[testfile]["p&r|{}|anomaly_vs_testfile".format(fuzzy_hash)] = "fp"

            # when the similarity score is = 0 and the file is labeled as normal (TN)
            elif sim_score == 0 and  file_label == "normal":
                tn_ctr += 1
                testfile_data_dict[testfile]["p&r|{}|anomaly_vs_testfile".format(fuzzy_hash)] = "tn"

            else: 
                print("ERROR, Similarity Score ", fuzzy_hash, " ", 
                    sim_score," label: " , testfile_data_dict[testfile]["label"])


    print(tp_ctr, fp_ctr, tn_ctr, fn_ctr)    


if __name__ == '__main__':
    config = read_yaml_file()
    path_to_testfiles = config["Path_to_testfiles"]
    algorithms = ["SSDEEP", "NILSIMSA", "MRSHV2"] #config["Fuzzy_hashing_algorithms"]

    hash_dict = create_evaluation_data_dict_from_testfiles(algorithms, path_to_testfiles, "normal")
    evaluation_data_dict = approximate_matching_anomaly_vs_files(hash_dict)
    #precision_and_recall_calculation(evaluation_data_dict)









