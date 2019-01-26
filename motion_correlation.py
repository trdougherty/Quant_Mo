# from __future__ import print_function
import numpy as np
import pandas as pd
import sys
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
import supplement as spp
from scipy import stats

## This is specific to this machine and experiment, should be replaced with config file if published
experiment = 'data/'
light_e = 'data/elles/data/'
light_h = 'data/light_filt/'
e_name = 'aw_motion'
l_nam = 'aw_light'
p_nam = 'all_data'
data = experiment+'data'
pickled = 'data/pickled'
photos = experiment+'photos'
filt = experiment+'filt'

for h in range(24):
    try:
        h = str(h)
        f_name = filt+'/'+h+e_name+'.npy'
        p_name = pickled+'/'+h+p_nam+'.pkl'
        l_name = light_h+h+l_nam+'.npy'
        print(p_name)

        M = np.load(f_name); # This should always work if the line before was run
        L = np.load(l_name); L = np.rot90(L); # This rot is to make the images follow the same pattern
        # This is looking at the raw influence of light on the motion of the frame
        light_red = gaussian_filter(np.rot90(L, k=3).astype(int), sigma=7)
        motion_red = np.sum(np.absolute(M), axis=2)
        # This is trimming our analysis to the good data
        front = 0.3; back = 0.74
        light_red = spp.edge(light_red, (front, back))
        motion_red = spp.edge(motion_red, (front, back))

        # This is looking at how the change in values is related to the other values
        light_change = np.gradient(light_red)
        motion_change = np.gradient(motion_red)

        data_setup = {'light':light_red.flatten(), 'light_dx':light_change[0].flatten(), 'light_dy':light_change[1].flatten(), 'raw_motion':motion_red.flatten(), 'motion_dx':motion_change[0].flatten(), 'motion_dy':motion_change[1].flatten()}
        df = pd.DataFrame(data=data_setup)

        # This next line removes outliers
        df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
        df.to_pickle(p_name)

        df_corr = df.corr() # This is going to give us a correlation matrix to play with
        print(df_corr)
    except ValueError:
        pass

        #print(df_corr)
