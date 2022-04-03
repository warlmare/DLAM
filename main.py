
import os, random, pprint

DATASET_FILETYPE = "pdf"
DATASET_SIZE = 30000
SAMPLE_FILES_PATH = "../govdocs/all_files"


def file_select():
    '''selects DATASET_SIZE amount of files of DATASET_FILETYPE

    :param FILETYPE:
    :return: dict with paths to files
    '''
    filepaths = {}

    filenames = random.sample(os.listdir(SAMPLE_FILES_PATH), DATASET_SIZE)
    for fname in filenames:
        srcpath = os.path.join(SAMPLE_FILES_PATH, fname)
        filesize = os.path.getsize(srcpath)
        filepaths[srcpath] = filesize

    return filepaths


# Press Umschalt+F10 to execute it or replace it with your code.
if __name__ == '__main__':

    sample_files_paths = file_select()
    pprint.pprint(sample_files_paths)


