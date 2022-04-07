
import os, random, helper, plotille, numpy as np
import termplotlib as tpl
import matplotlib.pyplot as plt

DATASET_FILETYPE = "pdf"
DATASET_SIZE = 10000
SAMPLE_FILES_PATH = "../govdocs/all_files"
HASHING_ALGORITHM = "TLSH"


def get_sample_files():
    '''selects DATASET_SIZE amount of files of DATASET_FILETYPE

    :param FILETYPE:
    :return: dict with paths to files
    '''
    filepaths_ext = []

    for file in os.listdir(SAMPLE_FILES_PATH):
        if file.endswith(DATASET_FILETYPE):
            filepath = os.path.join(SAMPLE_FILES_PATH, file)
            filepaths_ext.append(filepath)


    filesizes = []
    sample = random.sample(filepaths_ext, DATASET_SIZE)
    for file in sample:
        filesizes.append(helper.getfilesize(file) / 1000000) # size in MB



    #converting array to numpy array
    filesizes = np.asarray(filesizes)
    dataset_size = np.sum(filesizes)
    MAX_FILESIZE = np.max(filesizes)
    print("-" * 50, "FILESIZE DISTRIBUTION IN SAMPLE", "-" * 50)
    print(plotille.histogram(filesizes, x_min=0, x_max=MAX_FILESIZE, y_min=0, y_max=DATASET_SIZE/2, X_label="Size in MB"))
    print("DATASET SIZE: ", dataset_size, " MB")
    print("-" * 131)


    filepaths = random.sample(filepaths_ext, DATASET_SIZE)
    return filepaths, MAX_FILESIZE


def get_rand_bytes(byte_length: int):
    '''generates a random byte sequence of length byte_length

    :param byte_length:
    :return: byte sequence as byt object
    '''

    random_bytes = random.randbytes(byte_length)
    return random_bytes



# Press Umschalt+F10 to execute it or replace it with your code.
if __name__ == '__main__':

    #generating a sample_set of files
    sample_files_paths, max_filesize = get_sample_files()

    #creating a instance of a fuzzy hash
    algorithm_instance = helper.get_algorithm(HASHING_ALGORITHM)

    #hashing a file from the sample set
    sample_hash = algorithm_instance.get_hash(sample_files_paths[1])
    print(sample_hash)

    #generating a random sequence of bytes



