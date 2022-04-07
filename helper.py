import math
import time
import os
import itertools
import importlib
from pathlib import Path
import pandas as pd
import ntpath

def gettimestring():
    '''gets the current time as %Y%m%d-%H%M%S string

    :return: datetime as string
    '''
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return timestr

def getfilesize(filepath):
    '''gets total filesize in bytes

    :param filepath:

    :return: int filesize
    '''
    filesize = os.stat(filepath).st_size
    return filesize

def get_algorithm(algorithm: str):
    """creates a specific algorithm object.

    :param algorithm: string name of the Algorithm as spelled as classname
    :return: algorithm_obj
    """
    class_ = getattr(importlib.import_module("algorithms"), algorithm)
    instance = class_()
    return instance

def is_overlaping(a, b) -> bool:
    '''helper function

    :param a:
    :param b:
    :return: bool
    '''

    if b[0] > a[0] and b[0] < a[1]:
        return True
    else:
        return False

def merge_overlapping_tuples_2(tuple_list) -> list:
    '''merges overlapping tuples in a list

    Exp: [[1,3], [2,6], [8,10]] -> [[1, 6], [8, 10]]

    :param tuple_list: list of tuples with potential overlap
    :return: merged_list
    '''
    # sort the intervals by its first value
    tuple_list.sort(key=lambda x: x[0])

    merged_list = []
    merged_list.append(tuple_list[0])
    for i in range(1, len(tuple_list)):
        pop_element = merged_list.pop()
        if is_overlaping(pop_element, tuple_list[i]):
            new_element = pop_element[0], max(pop_element[1], tuple_list[i][1])
            merged_list.append(new_element)
        else:
            merged_list.append(pop_element)
            merged_list.append(tuple_list[i])
    return merged_list

def merge_overlapping_tuples(tuple_list) -> list:

    tuple_list.sort(key=lambda interval: interval[0])
    merged = [tuple_list[0]]
    for current in tuple_list:
        previous = merged[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)

    return merged

def range_diff(r1, r2):
    s1, e1 = r1
    s2, e2 = r2
    endpoints = sorted((s1, s2, e1, e2))
    result = []
    if endpoints[0] == s1:
        result.append((endpoints[0], endpoints[1]))
    if endpoints[3] == e1:
        result.append((endpoints[2], endpoints[3]))
    return result

def multirange_diff(r1_list, r2_list):
    for r2 in r2_list:
        r1_list = list(itertools.chain(*[range_diff(r1, r2) for r1 in r1_list]))
    return r1_list

def tuple_reindex(tpl_lst, single_tpl) -> list:
    a,b  = single_tpl[0]
    length = b - a

    for index, tuple in enumerate(tpl_lst):
        start = tuple[0]
        end = tuple[1]
        # if the first element in the tuple is bigger than the index point ...
        if start > index:
            start -= length
            if start < 0:
                start = 0
            end -= length
            tpl_lst[index]=(start, end)

    return tpl_lst

def get_tpl_lst(lst):
    '''converts list of lists into list of tuples

    :param lst:
    :return: list of tuples [(x,y),(x,y)]
    '''

    tuples_lst = [tuple(x) for x in lst]
    return tuples_lst

def get_lst_lst(tpl_lst):
    '''converts list of tuples into list of lists

    :param tpl_lst: list of tuples [(x,y),(x,y)]
    :return: list of lists [[x,y],[x,y]]
    '''
    lst_of_lst = [list(x) for x in tpl_lst]
    return lst_of_lst


def delete_overlapping_tuples_reindex(lst_a, lst_b) -> list:
    '''wrapper function that deletes a range (tuple) from a list
    of tuples and reindexes the tuples.

    any overlapping tuple in lst_a is adjusted to exclude the lst_b tuple
    the tuples in lst_a are then reindexed. So the range in tuple lst_b is
    effectivly deleted and closed in lst_a.

    lst_a = [[0,3],[4,7],[9,1]]
    lst_b = [[2,8]]

    --> [[0, 2], [3, 5]]

    :param lst_a:
    :param lst_b: single tuple list
    :return: conv_list , reconverted_list
    '''

    list_a = get_tpl_lst(lst_a)
    list_b = get_tpl_lst(lst_b)
    diff_lst = multirange_diff(list_a,list_b)
    reindex_diff_lst = tuple_reindex(diff_lst,list_b)
    conv_lst = get_lst_lst(reindex_diff_lst)

    return conv_lst

def placebo_insert_reindex(lst_a, lst_b) -> list:
    '''
    reindexes a tuple as if a tuple was inserted. No actual tuple is inserted.

    :param lst_a:
    :param lst_b: single tuple list
    :return: list
    '''
    #convert list of lists to tuples
    lst_a_conv = lst_a #get_tpl_lst(lst_a)
    lst_b_conv = lst_b #get_tpl_lst(lst_b)

    a,b = lst_b_conv[0]
    length = b - a

    for index, ele in enumerate(lst_a_conv):
        start = ele[0]
        end = ele[1]
        # if the first element in the tuple is bigger than the index point ...
        if start < a < end:
            lst_a_conv[index]=[start, a]
            new_tuple = [b, (end-a)+b]
            lst_a_conv.append(new_tuple)

        elif start > a and end > b and start is not b:
            start += length
            end += length
            lst_a_conv[index] = [start, end]

    lst_a_conv.sort(key=lambda x: x[0])
    reindexed_list = get_lst_lst(lst_a_conv)
    return reindexed_list


def insert_reindex(lst_a:list,lst_b:list) -> list:
    '''
    inserts a missing tuple, or merges it with others if overlapping.

    TODO: This abomination needs to be optimized

    :param lst_a: is the list that will be updated
    :param lst_b: is the list(list) that will be inserted
    :return: lst
    '''

    a, b = lst_b[0]
    length = b - a

    # if lst_a is empty
    if not lst_a:
        lst_a.append(lst_b[0])
        return lst_a


    for index, ele in enumerate(lst_a):
        start = ele[0]
        end = ele[1]

        # if start is smaller than a and end is bigger then enlarge the existing tuple
        if start < a < end < b:
            lst_a[index] = [start, b]
        elif a < start < b < end:
            lst_a[index] = [a, end]
        elif a < start:
            end += length
            start += length
            lst_a[index] = [start,end]
        elif abs(end - a) <= 1:
            lst_a[index] = [start,b]

    # all overlaps have been merged and all indexes have been reset in the previous loop
    for index, ele in enumerate(lst_a):
        start = ele[0]
        end = ele[1]

        if end < a:
            continue
        elif start < a and end == b:
            break
        elif start > b:
            lst_a.append(lst_b[0])
            break

    lst_a.sort(key=lambda x: x[0])

    return lst_a

def get_missing(l, start, stop):
    '''
    iterates over list of tuples an calculates the missing intervals

    :param l: list with current sequences
    :param start: beginning of sequence calculation
    :param stop: end of sequence calculation
    :return:
    '''
    new_list = []
    for tup in l:
        if tup[0] > start:
            new_list.append((start, tup[0]))
        start = tup[1]
    # add any left over values
    if start < stop:
        new_list.append((start, stop))

    new_list = get_lst_lst(new_list)

    return new_list

def get_indexed_tuples_list(lst) -> list:
    '''takes a list of tuples [[x,y],[y,z],...] and returns a indexed list [[0,x,y],[1,y,z],...]

    :param lst:
    :return: list
    '''
    new_list = []

    for pos, tup in enumerate(lst):
        start = tup[0]
        end = tup[1]

        new_list.append([pos, start, end])

    return new_list

def merge_adjacent_tuples(done, rest) -> list:
    '''merges adjacent tuples

    call via parse(a[:1], a[1:]) a beeing the list of lists

    a = [[0, 3], [4, 6], [6, 9], [9, 11], [11, 15], [15, 20]] --> [[0, 3], [6, 20]

    :param lst:
    :return: lst
    '''

    if len(rest) == 0:
        return (done)

    x = done.pop(-1)
    y = rest.pop(0)
    if x[1] == y[0]:
        done.append([x[0], y[1]])
    else:
        done.append(x)
        done.append(y)

    return merge_adjacent_tuples(done, rest)

def get_dataframe(list_of_lists:list) ->pd.DataFrame:
    '''takes a list of lists and turns it into a dataframe

    list of lists structure: [["column1","column2", ....],
                              [value1, value2]]

    first row is gonna be turned into the columns of the
    dataframe.

    :param list_of_lists:
    :return:
    '''

    df = pd.DataFrame(list_of_lists[1:], columns=list_of_lists[0])
    return df


def get_file_name(path: str) -> str:
    '''takes a filepath and returns the filename at the end

    :param path: path to file
    :return: filename
    '''
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_extensions_in_dir(dir_path)->list:
    '''returns a list with all filetypes in a dir

    :param dir_path:
    :return:
    '''
    excluded = ['', '.txt', '.lnk']
    extensions = set()
    for _, _, files in os.walk(dir_path):
        for f in files:
            ext = Path(f).suffix.lower()
            if not ext in excluded:
                extensions.add(ext)
    return extensions











