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

#Variables of interest
axis = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False,
	help="path to output video file")
args = vars(ap.parse_args())

def process(file):
    numObj = np.load(file)
    [date, iteration, ar1, ar2] = numObj
    recompose = np.zeros([ar1.shape[0],ar1.shape[1],2])
    recompose[:,:,0] = ar1
    recompose[:,:,1] = ar2
    return [date, iteration, recompose]

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

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(xx, yy, Z, arr.shape[0], cmap='binary')
    ax.set_title('{} Axis'.format(axis))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return


# Shouldn't really be needed
def normalize(x):
    return x/np.amax(np.absolute(x))

def saveArr(x):
    assert x.shape[2] == 2 #Test to verify we're passing in the correct motion vectors
    t = datetime.datetime.now().timestamp()
    arrayStorage = np.array([t,x[:,:,0],x[:,:,1]], dtype = object)
    np.save(args["output"], arrayStorage)


if __name__ == '__main__':
    files = fileImport()
    dates = []
    [date, sumIter, sumArray] = process(files[0]) #This gives us our first iterations and sum array
    dates.append(date)

    for i in files[1:]:
        [temp_date, ite, arr] = process(i)
        dates.append(temp_date)
        sumIter += ite
        sumArray += arr

    motionArr = sumArray / sumIter #KEY LINE

    # Allows us to work with the shape off the photo we're looking at
    printArr(motionArr, axis)

    #Finally saves the array
    saveArr(motionArr)
