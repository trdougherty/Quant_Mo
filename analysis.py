from __future__ import print_function
import time
import cv2
from OpticalFlowShowcase import *
import numpy as np
import argparse
import io
import sys
import datetime
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from memory_profiler import profile
from uncertainties import unumpy
import uncertainties as u
import ucert
import gc

# print(resource.getrlimit(resource.RLIMIT_STACK))
# print(sys.getrecursionlimit())

# max_rec = 0x100000

# # May segfault without this line. 0x100 is a guess at the size of each stack frame.
# resource.setrlimit(resource.RLIMIT_STACK, [0x100 * max_rec, resource.RLIM_INFINITY])
# sys.setrecursionlimit(max_rec)

#Variables of interest
## NOTE - 0 is X and 1 is Y
axis = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False,
	help="path to output video file"),
ap.add_argument("-s", "--save", required=False, dest='save', action='store_true',
    help="stores the array if provided"),
ap.add_argument("-diff", "--difference", required=False, dest='difference', action='store_true',
    help="yields discrepancy between arrays"),
ap.add_argument("-e", "--echo", required=False, dest='echo', action='store_true',
     help="prints sample output")
ap.set_defaults(save=False)
ap.set_defaults(difference=False)
ap.set_defaults(echo=False)
args = vars(ap.parse_args())

def process(file):
    numObj = np.load(file)
    [date, arr] = numObj
    return [date, arr]

# Not really needed for this
def localize(point, x, y, mv = 0.08):
    Z = 1/np.power(np.power(point[0]-x,2)+np.power(point[1]-y,2),0.5)
    Z[ Z > mv ] = mv
    return Z*(1/mv)

# This will give use the divergence of the array which we can use for localizing later
def gradient(array): return np.gradient(array) #np.add.reduce(np.gradient(array))

def humanDate(ts):
    return datetime.datetime.fromtimestamp(ts)

def sizeof_fmt(num, suffix='B'):
    ## Took this from online to read how much RAM this is using
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def fileImport():
    imported = sys.stdin.read()
    return imported.rstrip().split('\n')

def printArr(arr, axis):
    # Allows us to work with the shape off the photo we're looking at
    x_dist = np.arange(0,arr.shape[0])
    y_dist = np.arange(0,arr.shape[1])

    xx, yy = np.meshgrid(x_dist, y_dist)
    Z = arr[:,:,axis]

    min_ = -1; max_ = 1 # This is the default value
    if (Z.min() < -1): min_ = Z.min()
    if (Z.max() > 1): max_ = Z.max()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(xx, yy, Z, arr.shape[0], cmap='binary')
    ax.set_title('{} Axis'.format(axis))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_zlim(min_,max_)
    plt.show()
    return


# Shouldn't really be needed

def normalize(x):
    return x/np.amax(np.absolute(x))

def saveArr(x):
    assert x.shape[2] == 2 #Test to verify we're passing in the correct motion vectors
    t = datetime.datetime.now().timestamp()
    arrayStorage = np.array([t,x.copy()], dtype = object)
    print('Saving an array of size: {}'.format(sizeof_fmt(sys.getsizeof(arrayStorage))))
    np.save(args["output"], arrayStorage)

def intensity(x):
    # This is taking the hypotenuse of the intensity vectors
    assert type(x).__module__ == np.__name__
    return np.power(np.sum(np.power(x,2),axis=len(x.shape)-1),0.5) # pythagorean

def reshapeHelp(arr):
    ### This function converts the array into an added dimension to support concatenation later
    assert type(arr).__module__ == np.__name__
    lis = list(arr.shape)
    lis.insert(0,1)
    return np.reshape(arr,tuple(lis))


if __name__ == '__main__':
    diff = args["difference"]
    files = fileImport()
    print("Num of files:\t{}".format(len(files)))
    dates = []
    tempArr = np.array([],dtype=int)

    print("Output is:\t{}".format(args["output"]))

    if diff:
        assert len(files) >= 2
        [a_date, a_arr] = np.load(files[0])
        [b_date, b_arr] = np.load(files[1])
        u_array = b_arr - a_arr
    else:
        for i in files:
            [temp_date, arr] = process(i)
            dates.append(temp_date)
            if sys.getsizeof(tempArr) == 96:
                tempArr = reshapeHelp(arr)
            else:
                tempIn = reshapeHelp(arr)
                tempArr = np.concatenate((tempArr,tempIn),axis=0)
            
            print('Shape of the array is: {}'.format(tempArr.shape))
            print('Size of the array is: {}\n'.format(sizeof_fmt(sys.getsizeof(tempArr))))

        u_array = np.mean(tempArr,axis=0)

        # Try and mitigate the memory consumption
        u_array = u_array.copy()
        del tempArr
        gc.collect()

        print('Size of the final average: {}\n'.format(sizeof_fmt(sys.getsizeof(u_array))))
        print(u_array.shape)

    # Allows us to work with the shape off the photo we're looking at
    echo = args["echo"]
    if echo:
        printArr(normalize(unumpy.std_devs(u_array)), 0)
        printArr(normalize(unumpy.std_devs(u_array)), 1)

    # Finally saves the array
    save = args["save"]
    if save:
        saveArr(u_array)
