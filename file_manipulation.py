import os
import random
import helper


# TODO: rename chunk into block

# TODO: return-type annotation missing.

# TODO: This method needs to be transferred to helper.py
def getfilesize(filepath):
    '''gets total filesize in bytes
    :param filepath: path of file

    :return: int filesize
    '''
    filesize = os.stat(filepath).st_size
    return filesize


# TODO: make maxlen optional
def getrandoffset(filepath, maxlen):
    '''returns a random offset before EOF

    :param filepath:
    :param maxlen: largest possible offset (file len - maxlen)

    :return int offset
    '''
    filesize = getfilesize(filepath)
    maxoffset = filesize - maxlen
    offset = random.randrange(0, maxoffset)
    return offset


def getrandchunk(filepath, chunkSize):
    ''' gets random chunk of a certain size from a file

    :param filepath: path of the file
    :param chunkSize: size in bytes

    :return chunk in bytes
    '''

    offset = getrandoffset(filepath, chunkSize)

    buff = bytearray(chunkSize)
    f = open(filepath, "rb")
    f.seek(offset)
    buff = f.read(chunkSize)
    f.close()

    return buff

def overwrite_with_chunk(filepath, chunk):
    '''cuts a file into two halfes, inserts a chunk at a random position
    and rejoines the two halfs.

    :param: filepath
    :param chunk byte

    :return: modified bytes

    :raise: TODO: raise exception when writing EOF
    '''

    chunkSize = len(chunk)
    offset = getrandoffset(filepath, chunkSize)

    # end is where the chunk ends and the second half begins
    end = offset + chunkSize
    f = open(filepath, "rb")
    first_half = f.read(offset)
    f.seek(end)
    second_half = f.read()
    f.close()

    # merging the three parts
    byt = first_half + chunk + second_half

    return byt, offset

def overwrite_with_chunk_byt(filepath, chunk_byt):
    '''cuts a file into two halfes, inserts a chunk at a random position
    and rejoines the two halfs.

    :param: filepath
    :param chunk byte

    :return: modified bytes

    :raise: TODO: raise exception when writing EOF
    '''

    chunkSize = len(chunk_byt)
    offset = getrandoffset(filepath, chunkSize)

    # end is where the chunk ends and the second half begins
    end = offset + chunkSize
    f = open(filepath, "rb")
    first_half = f.read(offset)
    f.seek(end)
    second_half = f.read()
    f.close()

    # merging the three parts
    byt = first_half + chunk_byt + second_half

    return byt, offset

def common_block_insertion(filePath1, filePath2, chunk_filePath, chunksize):
    '''
    insert common block at random positions into two files

    :param filePath1:
    :param filePath2:
    :param chunk_filePath: Path to file of which parts will be inserted

    :return: two new byte arrays with a shared common block
    '''

    chunk = getrandchunk(chunk_filePath, chunksize)

    #the offsets that are passed here are not important in this method
    byt1, offset1 = overwrite_with_chunk(filePath1, chunk)
    byt2, offset2 = overwrite_with_chunk(filePath2, chunk)

    return byt1, byt2

def common_block_insertion_byt(filePath1, filePath2, chunk_bytes):
    '''
    insert common byte block at random positions into two files

    :param filePath1:
    :param filePath2:
    :param chunk_bytes: bytesobject that will be inserted in the file

    :return: two new byte arrays with a shared common block
    '''


    #the offsets that are passed here are not important in this method
    byt1, offset1 = overwrite_with_chunk_byt(filePath1, chunk_bytes)
    byt2, offset2 = overwrite_with_chunk_byt(filePath2, chunk_bytes)

    return byt1, byt2

def random_cutting_perc(filepath, percentage):
    '''
    cuts off a certain percentage from a file

    :param filepath:

    :return byt: bytestream of the newly created file
    '''
    filesize = getfilesize(filepath)
    cutoff = int((percentage * filesize) / 100.0)
    offset = getrandoffset(filepath, cutoff)

    # end is where the cutoff ends and the second half begins
    end = offset + cutoff
    f = open(filepath, "rb")

    # read until offset
    first_half = f.read(offset)

    # set pointer to end offset after the cutoff
    f.seek(end)
    second_half = f.read()
    f.close()

    # TODO: what if the cutoff is just in the end !?!? this needs to be caught
    # merging the two parts
    byt = first_half + second_half

    return byt


def random_cutting_byte(filepath, size:int):
    '''
    cuts off a certain amount of bytes from a file

    :param filepath:

    :return byt: bytestream of the newly created file
    '''


    offset = getrandoffset(filepath, size)

    # end is where the cutoff ends and the second half begins
    end = offset + size
    f = open(filepath, "rb")

    # read until offset
    first_half = f.read(offset)

    # set pointer to end offset after the cutoff
    f.seek(end)
    second_half = f.read()
    f.close()

    # TODO: what if the cutoff is just in the end !?!? this needs to be caught
    byt = first_half + second_half

    return byt, offset


def end_side_cutting(filepath, percentage):
    '''
     cuts off a certain percentage from the tail of a file

    :param filepath:
    :param percentage:

    :return byt: the bytes of the file - cutoff
    '''

    filesize = getfilesize(filepath)
    cutoff = int((percentage * filesize) / 100.0)

    # file - cutoff
    first_part_len = filesize - cutoff

    # TODO: this could be done with truncate more easily
    f = open(filepath, "rb")
    byt = f.read(first_part_len)
    f.close()

    return byt

    # TODO: check wether the remaining file has the right byte size


def front_side_cutting(filepath, percentage):
    '''
     cuts off a certain percentage from the tail of a file

    :param filepath:
    :param percentage:

    :return byt: the bytes of the file - cutoff
    '''

    filesize = getfilesize(filepath)
    cutoff = int((percentage * filesize) / 100.0)

    f = open(filepath, "rb")
    #jumps to the offset after the cutoff
    f.seek(cutoff)
    #reads everything after the offset from file
    byt = f.read()
    f.close()

    return byt

    # TODO: check wether the remaining file has the right byte size
def front_side_cutting_byte(bytes_obj, percentage):
    '''
     cuts off a certain percentage from the tail of a byte object

    :param bytes_obj:
    :param percentage:

    :return byt: the bytes of the file - cutoff
    '''

    filesize = len(bytes_obj)
    cutoff = int((percentage * filesize) / 100.0)
    return bytes_obj[cutoff:]



def random_byte_generation(size):
    '''generates a random set of bytes


    :param size: int (number of bytes)
    :return block: byte (the generated block in bytes)
    '''

    random_bytes = random.randbytes(size)
    return random_bytes

def get_random_files(directoy_path, size, file_ctr):
    ''' creates file_ctr amount of files of size (size)
    that are made up of totally random bytes

    :param directoy_path:
    :param size:
    :paran file_ctr:
    '''

    for x in range(file_ctr):
        byt = random_byte_generation(size)
        filename = str(size) + "_" + str(x)
        filepath = directoy_path + "/" + filename
        f = open(filepath, "wb")
        f.write(byt)
        f.close()


def percentage_blocks_random_head_overwrite(filepath, percentage):
    '''
    inserts a percentage of random bytes at the beginning of a file

    :param filepath:
    :param percentage: this amount of the file will be overwritten with random bytes
    '''

    filesize = getfilesize(filepath)

    offset = int((percentage * filesize) / 100.0)

    f = open(filepath, "rb")

    # generate the first randomly generated half of the file
    first_half = random_byte_generation(offset)

    # move pointer to offset and read the second half of the file from the offset
    f.seek(offset)
    second_half = f.read()
    f.close()

    # combining the new first half and the old second half
    byt = first_half + second_half

    return byt

def percentage_blocks_random_tail_overwrite(filepath, percentage):
    '''
    inserts a percentage of random bytes at the beginning of a file

    :param filepath:
    :param percentage: this amount of the file will be overwritten with random bytes
    '''

    filesize = getfilesize(filepath)

    offset = int((percentage * filesize) / 100.0)

    f = open(filepath, "rb")

    # generate the second randomly generated half of the file
    second_half = random_byte_generation(offset)

    #reads file until offset -> first half of file
    first_half = f.read(offset)
    f.close()

    # combining the new first half and the old second half
    byt = first_half + second_half

    return byt

def percentage_blocks_random_head_tail_insert(filepath, mode, percentage):
    '''
    inserts a percentage of random bytes at the beginning of a file

    :param filepath:
    :param mode: head or tail insert
    :param percentage: this amount of the file will be overwritten with random bytes
    '''

    filesize = helper.getfilesize(filepath)
    quotient = 0.01 * percentage

    offset = int(quotient * filesize)


    # generate the first randomly generated half of the file
    random_part = random_byte_generation(offset)

    # move pointer to offset and read the second half of the file from the offset
    f = open(filepath, "rb")
    file_part = f.read()
    f.close()

    # combining the new first half and the old second half
    if mode == "percentage_head":
        byt = random_part + file_part
    elif mode == "percentage_tail":
        byt =  file_part + random_part
    return byt

def fixed_blocks_random_head_overwrite(filepath, blocklength):
    '''overrides the head of the file with specified amount of random bytes

    :param filepath:
    :param blocklength: random bytes to be written at the beginning of the file
    :return byt: the manipulated file copy as bytes
    '''

    filesize = getfilesize(filepath)
    f = open(filepath, "rb")

    # generate the first randomly generated half of the file
    first_half = random_byte_generation(blocklength)

    # move pointer to offset and read the second half of the file from the offset
    f.seek(blocklength)
    second_half = f.read()
    f.close()

    # combining the new first half and the old second half
    byt = first_half + second_half

    return byt

def fixed_blocks_random_head_insert(filepath, blocklength):
    '''overrides the head of the file with specified amount of random bytes

    :param filepath:
    :param blocklength: random bytes to be written at the beginning of the file
    :return byt: the manipulated file copy as bytes
    '''

    filesize = getfilesize(filepath)
    f = open(filepath, "rb")

    # generate the first randomly generated half of the file
    first_half = random_byte_generation(blocklength)

    # move pointer to offset and read the second half of the file from the offset

    second_half = f.read()
    f.close()

    # combining the new first half and the old second half
    byt = first_half + second_half

    return byt


def random_substitution(filepath:str, blocklength:int) -> bytes:
    '''
    inserts random bytes at a random position in file

    :param filepath: copy of this file will be manipulated
    :param blocklength: this amount of random bytes will be substituted
    :return byt: the manipulated file copy as bytes
    '''

    chunk = random_byte_generation(blocklength)
    byt, offset = overwrite_with_chunk(filepath, chunk)
    return byt, offset


# TODO: Still overwrites and does not insert
def random_insertion(filepath:str, blocklength:int) -> bytes:
    '''
    inserts random bytes of blocklength at a random position in a file.

    :param filepath: copy of this file will be manipulated
    :param blocklength: amount of random bytes that will be inserted
    :return byt: the manipulated file copy as bytes
    '''

    chunk = random_byte_generation(blocklength)
    byt, offset = overwrite_with_chunk(filepath, chunk)

    return byt, offset



if __name__ == '__main__':

    filePath1 = "../../testdata/2048/test_file1_2048"
    filePath2 = "../../testdata/2048/test_file2_2048"
    chunk_filePath = "../../testdata/test_file1"

    #byt1 ,byt2 = common_block_insertion(filePath1, filePath2, chunk_filePath, 1600)


def get_rand_chunk_of_ext(ext, chunk_size):
    '''given a extension, this method select a specified amount of bytes from a
    file with that extension

    :param ext:
    :return:
    '''
    # chunkfile is the file that will be inserted
    for file in os.listdir("../../testdata/filetype_testfiles"):
        if file.endswith(ext):
            chunkfile_path = os.path.join("../../testdata/filetype_testfiles", file)

    rand_chunk_ext = getrandchunk(chunkfile_path,chunk_size)
    return rand_chunk_ext