import os
import helper

dir_path = "../mixed_files/"
tlsh_instance = helper.get_algorithm("TLSH")

for file in os.listdir(dir_path):
    try:
        hash_a = tlsh_instance.get_hash(dir_path + file)
    except ValueError:
        print(file)
        os.remove(dir_path + file)

