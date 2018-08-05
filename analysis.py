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

#Variables of interest
axis = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False,
	help="path to output video file"),
ap.add_argument("-diff", "--difference", required=False, dest='difference', action='store_true',
    help="yields discrepancy between arrays")
 ap.add_argument("-p", "--print", required=False, dest='p', action='store_true',
     help="prints sample output")
ap.set_defaults(difference=False)
args = vars(ap.parse_args())

def process(file):
    numObj = np.load(file, encoding = 'latin1')
    [date, iteration, ar1, ar2] = numObj
    recompose = np.zeros([ar1.shape[0],ar1.shape[1],2])
    recompose[:,:,0] = ar1
    recompose[:,:,1] = ar2
    return [date, iteration, recompose]

def f_process(file):
    numObj = np.load(file)
    [date, ar1, ar2] = numObj
    recompose = np.zeros([ar1.shape[0],ar1.shape[1],2])
    recompose[:,:,0] = ar1
    recompose[:,:,1] = ar2
    return [date, recompose]

# Not really needed for this
def localize(point, x, y, mv = 0.08):
    Z = 1/np.power(np.power(point[0]-x,2)+np.power(point[1]-y,2),0.5)
    Z[ Z > mv ] = mv
    return Z*(1/mv)

# This will give use the divergence of the array which we can use for localizing later
def gradient(array): return np.gradient(array) #np.add.reduce(np.gradient(array))

def humanDate(ts):
    return datetime.datetime.fromtimestamp(ts)

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

@profile
def uncert(arr):
    assert type(arr).__module__ == np.__name__ # This confirms we're working with a numpy array
    assert len(arr.shape) >= 3 # This confirms we're working with at least a 4D array
    a = unumpy.uarray(np.sum(arr,axis=0)/arr.shape[0],np.std(arr,axis=0))
    return a


def saveArr(x):
    assert x.shape[2] == 2 #Test to verify we're passing in the correct motion vectors
    t = datetime.datetime.now().timestamp()
    arrayStorage = np.array([t,x], dtype = object)
    np.save(args["output"], arrayStorage)


if __name__ == '__main__':
    diff = args["difference"]
    files = fileImport()
    print("Num of files:\t{}".format(len(files)))
    dates = []
    tempArr = []

    print("Output is:\t{}".format(args["output"]))

    if diff:
        assert len(files) >= 2
        [a_date, a_arr] = f_process(files[0])
        [b_date, b_arr] = f_process(files[1])
        motionArr = b_arr - a_arr
    else:
        for i in files:
            [temp_date, ite, arr] = process(i)
            dates.append(temp_date)
            tempArr.append(arr)

        totArr = np.array(tempArr, dtype='uint8')
        u_array = uncert(totArr)

    # Allows us to work with the shape off the photo we're looking at
    printArr(unumpy.std_devs(u_array), axis)

    # Finally saves the array
    saveArr(motionArr)
