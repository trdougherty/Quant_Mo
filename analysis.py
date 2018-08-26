from __future__ import print_function
import time
import numpy as np
import argparse
import io
import sys
import datetime
import ucert
import os

# print(resource.getrlimit(resource.RLIMIT_STACK))
# print(sys.getrecursionlimit())

# max_rec = 0x100000

# # May segfault without this line. 0x100 is a guess at the size of each stack frame.
# resource.setrlimit(resource.RLIMIT_STACK, [0x100 * max_rec, resource.RLIM_INFINITY])
# sys.setrecursionlimit(max_rec)

# Variables of interest
## NOTE - 0 is X and 1 is Y
axis = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False, dest='input',
                help="full path to the location of files"),
ap.add_argument("-o", "--output", required=False, dest='output',
                help="path to output video file"),
ap.add_argument("-diff", "--difference", required=False, dest='difference', action='store_true',
                help="yields discrepancy between arrays"),
ap.set_defaults(input='.')
ap.set_defaults(output='.')
ap.set_defaults(difference=False)
args = vars(ap.parse_args())

def process(file):
    try:
        numObj = np.load(file)
        [date, arr] = numObj
        return [date, arr]
    except EOFError:
        return []

def sizeof_fmt(num, suffix='B'):
    # Took this from online to read how much RAM this is using
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def absoluteFilePaths(directory):
    filenames = os.listdir(directory)
    return [directory+i for i in filenames if '.npy' in i ]

def saveArr(x):
    # Test to verify we're passing in the correct motion vectors
    assert x.shape[2] == 2
    t = datetime.datetime.now().timestamp()
    arrayStorage = np.array([t, x.copy()], dtype=object)
    print('Saving an array of size: {}'.format(
        sizeof_fmt(sys.getsizeof(arrayStorage))))
    np.save(args["output"], arrayStorage)

def saveTxt(u_array):
    with open(args["output"]+'.txt','w+') as outfile:
        for i in u_array:
            np.savetxt(outfile, i, fmt='%r')


def reshapeHelp(arr):
    # This function converts the array into an added dimension to support concatenation later
    assert type(arr).__module__ == np.__name__
    lis = list(arr.shape)
    lis.insert(0, 1)
    return np.reshape(arr, tuple(lis))


if __name__ == '__main__':
    diff = args["difference"]
    files = absoluteFilePaths(args["input"])
    assert files != None # To validate that we don't have a null array
    print("Num of files:\t{}".format(len(files)))
    dates = []
    tempArr = np.array([], dtype=int)

    print("Output is:\t{}".format(args["output"]))

    if diff:
        # This will need to be modified in the future
        assert len(files) >= 2
        [a_date, a_arr] = np.load(files[0])
        [b_date, b_arr] = np.load(files[1])
        u_array = b_arr - a_arr
    else:
        for i in files:
            temp = process(i)
            if temp:
                [temp_date, arr] = temp
                dates.append(temp_date)
                if sys.getsizeof(tempArr) == 96:
                    tempArr = reshapeHelp(arr)
                else:
                    tempIn = reshapeHelp(arr)
                    tempArr = np.concatenate((tempArr, tempIn), axis=0)

                print('Shape of the array is: {}'.format(tempArr.shape))
                print('Size of the array is: {}\n'.format(
                    sizeof_fmt(sys.getsizeof(tempArr))))

        u_array = np.mean(tempArr, axis=0)

    print('Size of the final average: {}\n'.format(sizeof_fmt(sys.getsizeof(u_array))))
    print(u_array.shape)

    saveTxt(u_array)
