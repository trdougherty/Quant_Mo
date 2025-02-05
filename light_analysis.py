from __future__ import print_function
import time
import numpy as np
from uncertainties import unumpy
import cv2
import argparse
import io
import sys
import datetime
import os
import pdb
from scipy import ndimage

# Variables of interest
## NOTE - 0 is X and 1 is Y
axis = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False, dest='input',
                help="full path to the location of files"),
ap.add_argument("-o", "--output", required=False, dest='output',
                help="path to output video file"),
ap.add_argument("-d", "--difference", required=False, dest='difference', action='store_true',
                help="yields discrepancy between arrays"),
ap.add_argument("-e", "--evolution", help="show the evolution of the system"),
ap.add_argument("-s", "--e_step", required=False, help="resolution of the average video", type=int, nargs='?',                        const=50, default=None),
ap.set_defaults(input='.')
ap.set_defaults(output='.')
ap.set_defaults(difference=False) 
ap.set_defaults(evolution="evolution")
args = vars(ap.parse_args())

def processLight(file):
    try:
        # This collapses the colors of the array into greyscale
        A = ndimage.imread(file)
        return np.sum(A, axis=len(A.shape)-1)
    except EOFError:
        return None

def process(file):
    try:
        # Assumes shape of (X,X,i) for this array - otherwise unumpy array would be unable to cope
        numObj = np.load(file)
        [date, arr] = numObj
        A = unumpy.matrix(arr.flatten())
        A_nom = np.ravel(A.nominal_values)
        outArr = np.reshape(A_nom, arr.shape)
        return [date, outArr]
    except EOFError:
        return None

def filterTime(files, pattern):
    return [ x for x in files if pattern in x ]

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

def edge(arr):
    lower_bound = 0.2; upper_bound = 0.8
    x_ = arr.shape[0]; y_ = arr.shape[1]
    return arr[int(x_*lower_bound):int(x_*upper_bound),int(y_*lower_bound):int(y_*upper_bound),:]

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
    return np.reshape(arr, tuple(lis)).astype('float16')

if __name__ == '__main__':
    diff = args["difference"]
    files = absoluteFilePaths(args["input"])
    assert files != None # To validate that we don't have a null array
    print("Num of files:\t{}".format(len(files)))
    dates = []
    iterat = 0
    tempArr = np.array([], dtype='float16')

    print("Output is:\t{}".format(args["output"]))

    if diff:
        # This will need to be modified in the future
        assert len(files) >= 2
        [a_date, a_arr] = np.load(files[0])
        [b_date, b_arr] = np.load(files[1])
        u_array = b_arr - a_arr
    else:
        for c,i in enumerate(files):
            temp = process(i)
            if temp:
                [temp_date, arr] = temp
                dates.append(temp_date)
                if sys.getsizeof(tempArr) == 96:
                    tempArr = reshapeHelp(arr)
                else:
                    tempArr = np.concatenate([tempArr, reshapeHelp(arr)], axis=0)

                print('SIZE: {}\n'.format(
                    sizeof_fmt(sys.getsizeof(tempArr))))
                print('SHAPE: {}'.format(tempArr.shape))

            if args["evolution"]: 
                evoPath = str(os.getcwd())+"/"+str(args["evolution"])
                if not os.path.exists(evoPath):
                    os.mkdir(evoPath)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
                out_y = cv2.VideoWriter(str(os.getcwd()+"/"+str(args["evolution"])+"/"+str(args["evolution"])+"_y"+".avi"), fourcc, 20.0, (temp[1].shape[0], temp[1].shape[1]))
                out_x = cv2.VideoWriter(str(os.getcwd()+"/"+str(args["evolution"])+"/"+str(args["evolution"])+"_x"+".avi"), fourcc, 20.0, (temp[1].shape[0], temp[1].shape[1]))

            if args["evolution"] and c%int(args["e_step"])==0:
                print("Saving evolution video")
                A = np.mean(tempArr, axis=0)
                A = A.astype(float)
                np.save(str(args["evolution"])+str(c), A)
                out_y.write(A[...,1])
                out_x.write(A[...,0])
            
        u_array = np.mean(tempArr, axis=0)
        u_array_std = np.std(tempArr, axis=0)
        print("\n-----------FINISHED PROCESSING-----------\n")

    print('FINAL VALUES: {}\n'.format(u_array))
    print('FINAL ERROR: {}'.format(u_array_std))
    print('FINAL SIZE: {}'.format(sizeof_fmt(sys.getsizeof(u_array))))
    print('FINAL SHAPE: {}\n'.format(u_array.shape))

    if args["evolution"]: out_y.release(); out_x.release();

    np.save(args["output"], u_array)
    np.save(args["output"]+"_std", u_array_std)
