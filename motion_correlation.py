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
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import scipy
import supplement as sp
import scipy.ndimage as ndimage

## This is specific to this machine and experiment, should be replaced with config file if published
experiment = '/Users/TRD/Research_Personal/Quant_Mo/data/aw_motion/'
light_e = '/Users/TRD/Research_Personal/Light-Barometer/data/elles/data/'
light_h = '/Users/TRD/Research_Personal/Light-Barometer/data/elles/filt/'
e_name = 'aw_motion'
l_nam = 'aw_light'
data = experiment + 'data'
pickled = experiment + 'pickled'
photos = experiment + 'photos'
filt = experiment + 'filt'

# Hour of interest here:
hour = 7
h = str(hour)
f_name = filt+'/'+h+e_name+'.npy'
l_name = light_h+h+l_nam+'.npy'
M = np.load(f_name); # This should always work if the line before was run
L = np.load(l_name)

# This is looking at the raw influence of light on the motion of the frame
data_setup = {'light':L.flatten(), 'motion':np.sum(np.absolute(M), axis=2).flatten()}
df = pd.DataFrame(data=data_setup)
if len(sys.argv) > 1:
    if str(sys.argv[1]) == '0':
        print(df.light.to_string(index=False))
    else:
        print(df.motion.to_string(index=False))
else:
    pass