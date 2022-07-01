
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


DATASET_FILETYPE = "js"
TRAINING_DATASET_SIZE = 100000 #govdocs currently holds 401498
TEST_DATASET_SIZE = 3000
TEST_DATA_FILES_PATH = "../napierone"
SAMPLE_FILES_PATH = "../javascript_corpus/data" #"../govdocs/all_files"
HASHING_ALGORITHM = "NILSIMSA"
MAX_FRAGMENT_PERCENTAGE = 99
MIN_FRAGMENT_PERCENTAGE = 1
FRAGMENT_VAR = False # if False than the same fragment is inserted into all files
FRAGMENT_SCRAMBLE_BOOL = False # if True than the fragment is rearranged 
FRAGMENT_SCRAMBLE_PERC = 0.10 # The anomalie will be cut in equal parts of size .._SCRAMBLE_PERC which will be mixed up 

def get_sample_files(tr_dataset_size, filedirectory):
    '''selects TRAINING_DATASET_SIZE amount of files of DATASET_FILETYPE

    :param FILETYPE:
    :return: dict with paths to files
    '''
    filepaths_ext = []

    if DATASET_FILETYPE == "all":
        for file in os.listdir(filedirectory):
            filepath = os.path.join(filedirectory, file)
            filepaths_ext.append(filepath)
    else:
        for file in os.listdir(filedirectory):
            if file.endswith(DATASET_FILETYPE):
                filepath = os.path.join(filedirectory, file)
                filepaths_ext.append(filepath)


    filesizes = []
    sample = random.sample(filepaths_ext, tr_dataset_size)
    for file in sample:
        filesizes.append(helper.getfilesize(file) / 1000000) # size in MB



    #converting array to numpy array
    filesizes = np.asarray(filesizes    )   #ataset_size = np.sum(filesizes)
    MAX_FILESIZE = np.max(filesizes) * 1000000
    #print("-" * 50, "FILESIZE DISTRIBUTION IN SAMPLE", "-" * 50)
    #print(plotille.histogram(filesizes, x_min=0, x_max=MAX_FILESIZE, y_min=0, y_max=TRAINING_DATASET_SIZE/2, X_label="Size in MB"))
    #print("DATASET SIZE: ", TRAINING_DATASET_SIZE, " MB")
    #print("-" * 131)


    filepaths_all = random.sample(filepaths_ext, tr_dataset_size)
    training_files = random.sample(filepaths_all, tr_dataset_size)
    # stores the difference of the list for training files and all files
    #rest_list = []

    #calculate the difference between the files already picked for trainging and all files 
    # in order to pick only so far un-picked files for testing
    #rest_list = list(set(filepaths_all) - set(training_files))
    #test_files = random.sample(rest_list, test_dataset_size)
    return training_files , int(MAX_FILESIZE)


def get_rand_bytes(byte_length: int):
    '''generates a random byte sequence of length byte_length

    :param byte_length:
    :return: byte sequence as byt object
    '''

    random_bytes = os.urandom(byte_length) 
       
    return random_bytes


def overwrite_with_chunk(filepath, fragment, fragment_size_percent):
    '''takes a random generated byte and a filepath.
    It insers fragment_size_percent (%) into a file (filepath)
    and copies the finished into a specified directory path.

    :param filepath:
    :param fragment:
    :return:
    '''


    filesize = helper.getfilesize(filepath)

    #calculates how long the fragment should be
    fragment_len = int(filesize * (fragment_size_percent * 0.01))

    #choose random position in byte object and cut fragment
    full_fragment_len = len(fragment)
    max_offset = full_fragment_len - fragment_len
    #print(max_offset)
    fragment_start_pos = 0 #random.randrange(0, max_offset)
    fragment_stop_pos = fragment_start_pos + fragment_len

    #choose random offset in file
    offset = file_manipulation.getrandoffset(filepath, fragment_len)

    #cut fragment_len from random pos in file
    fragment_ins = fragment[fragment_start_pos:fragment_stop_pos]

    # if 

    # end is where the chunk ends and the second half begins
    end = offset + fragment_len
    f = open(filepath, "rb")
    first_half = f.read(offset)
    f.seek(end)
    second_half = f.read()
    f.close()

    # merging the three parts
    byt = first_half + fragment_ins + second_half

    # writing bytes to file (filepath)
    filename = helper.get_file_name(filepath)
    f = open("./dataset/anomalies/{}".format(filename), "wb")
    f.write(byt)
    f.close()

    fragment_filepath = "./dataset/anomalies/{}".format(filename)

    return fragment_filepath


def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

def generate_dataset(training_dataset_size, path, min_fragment_size, max_fragment_size, generate_new_anomaly_flag):

    # calculating how many files have to be injected with an anomaly
    anomalous_files = int(TRAINING_DATASET_SIZE / 2)

    #generating a sample_set of files
    sample_files_paths, max_filesize = get_sample_files(training_dataset_size, path)

    #dividing the dataset into two parts
    list_a, normal_files = split_list(sample_files_paths)

    # if the fragment is supposed to be the same fragment for every new insertion
    if FRAGMENT_VAR is False:
        
        # if generating a new anomaly is wished otherwise read an exisiting one
        if generate_new_anomaly_flag is True:
            anomaly = get_rand_bytes(max_filesize)

            # creating a file with only the randomly generated files
            f = open("./dataset/anomaly", "wb")
            f.write(anomaly)
            f.close()
        else:
            f = open("./dataset/anomaly", "rb")
            anomaly = f.read()
            f.close()

    anomaly_files = []

    ctr = 0
    for path in list_a:
        # if the fragment is supposed to be varied than generate a new fragment for every new insertion
        file_size = file_manipulation.getfilesize(path)
        if FRAGMENT_VAR is True:
            anomaly = get_rand_bytes(file_size)

        if FRAGMENT_SCRAMBLE_BOOL is True:
            # if the fragment is supposed to be polymorphus i. e. be cut up in equal parts of x that are rearranged

            chunk_size = int(len(anomaly) * FRAGMENT_SCRAMBLE_PERC)
            #list[bytes] is created
            bytes_list = [anomaly[i:i+chunk_size] for i in range(0, len(anomaly), chunk_size)]

            # the list[bytes] of the anomaly are shuffled 
            bytes_list_shuffled = random.sample(bytes_list, len(bytes_list))
            anomaly = b''.join(bytes_list_shuffled)


        #insert fragments into file, the 
        fragment_size = random.randint(min_fragment_size,max_fragment_size)
        fragment_filepath = overwrite_with_chunk(path, anomaly, fragment_size)
        anomaly_files.append(fragment_filepath)
        #print("DATASET GENERATION: {}/{}".format(ctr,TRAINING_DATASET_SIZE))
        ctr += 1
        print(f"anomalous files created= {ctr}/{anomalous_files}")



    return anomaly_files, normal_files

def generate_hashes_from_dataset(dataset_paths):

    hashes = []

    algorithm_instance = helper.get_algorithm(HASHING_ALGORITHM)
    file_count = len(dataset_paths)

    ctr = 0
    for path in dataset_paths:
        sample_hash = algorithm_instance.get_hash(path)
        hashes.append(sample_hash)
    
        ctr += 1 
        print(f"files hashed= {ctr}/{file_count}")

    return hashes

def list_to_csv(list_x, filename):
    with open(filename, 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(list_x)




if __name__ == '__main__':

    train_dataset_split = int(TRAINING_DATASET_SIZE / 2)

    training_anomaly_files, training_normal_files = generate_dataset(TRAINING_DATASET_SIZE, SAMPLE_FILES_PATH, MIN_FRAGMENT_PERCENTAGE, MAX_FRAGMENT_PERCENTAGE, True)
    training_anomaly_hashes = generate_hashes_from_dataset(training_anomaly_files)
    list_to_csv(training_anomaly_hashes, "dataset/anomaly_hashes_{}_singlefragment_1-99_js_nilsimsa_jscorpus.csv".format(train_dataset_split))
    training_normal_hashes = generate_hashes_from_dataset(training_normal_files)
    list_to_csv(training_normal_hashes,  "dataset/normal_hashes_{}_singlefragment_1-99_js_nilsimsa_jscorpus.csv".format(train_dataset_split))

    test_dataset_split = int(TEST_DATASET_SIZE / 2)

    test_anomaly_files, test_normal_files = generate_dataset(TEST_DATASET_SIZE, TEST_DATA_FILES_PATH, 5, 15, False)
    test_anomaly_hashes = generate_hashes_from_dataset(test_anomaly_files)
    test_normal_hashes = generate_hashes_from_dataset(test_normal_files)
    list_to_csv(test_anomaly_hashes, "dataset/anomaly_hashes_{}_singlefragment_5_15_perc_js_nilsimsa_napierone.csv".format(test_dataset_split))
    list_to_csv(test_normal_hashes, "dataset/normal_hashes_{}_singlefragment_5_15_perc_js_nilsimsa_napierone.csv".format(test_dataset_split))

# to check filetypes and counts find . -type f | sed -n 's/..*\.//p' | sort | uniq -c
# find . -type f -size -50c

# delete file smaller than 50 bytes
#find . -type f -size -50c -delete


# count files with a particular extension 
#ls -lR /path/to/dir/*.jpg | wc -l



#find dataset/anomalies/ -name '*.js' | xargs rm

