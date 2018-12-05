print('stage1')

# from __future__ import print_function
import time
import numpy as np
import argparse
import io
import os
import sys
import datetime
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
from uncertainties import unumpy
import uncertainties as u
import itertools
import ucert
import seaborn as sns
import pandas as pd
from tqdm import tqdm

print('stage2')

experiment = os.getcwd()+ '/data/aw_motion/'
data = experiment + 'data'
pickled = experiment + 'pickled'
photos = experiment + 'photos'

def proc(file):     
    numObj = np.load(file)
    [date, arr] = numObj
    A = unumpy.matrix(arr.flatten())
    # this SHOULD build a new array that's twice as big -
    # but everything is float values instead of strings (yay) and alternates
    # between nominal, std, nominal, std
    A_nom = np.ravel(A.nominal_values)
    outArr = np.reshape(A_nom, arr.shape)
    return [date, outArr]


print('this was run')
if os.path.exists(pickled) is False:
    os.makedirs(pickled)
if os.path.exists(photos) is False:
    os.makedirs(photos)

print("Pickled Location is: {}".format(pickled))

lt_motion = pd.DataFrame([])
arr = ['Date','X','Y','xy','m']
lt_motion = pd.DataFrame(columns=arr)
for root, dirs, files in os.walk(data):
    motion_files = files

#print(lt_motion)
for c,i in enumerate(motion_files):
    print("On file:\t{} of total:\t{}".format(c, len(motion_files)))
    mot = proc(data+'/'+i) #opens the data
    x, y, z = mot[1].shape # Gives us the shape of the object
    mot_date =  pd.to_datetime(mot[0], format='%Y-%m-%d %H:%M:%S.%f') # Gives us the dateimte
    for i in range(x):
        for j in range(y):
            for k in range(z):
                lt_motion.loc[c*x*y*z+i*y*z+j*y+k] = [mot_date,i,j,k,mot[1][i][j][k]]

lt_motion.to_pickle(pickled+'/aw_motion.pkl')