# from __future__ import print_function
import numpy as np
import pandas as pd
import sys
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
import supplement as spp
from scipy import stats
import os
import light_analysis as la

## This is specific to this machine and experiment, should be replaced with config file if published
experiment = 'data/'
light_e = 'data/elles/data/'
light_h = 'data/light_filt/'
e_name = 'aw_motion'
l_nam = 'aw_light'
p_nam = 'long_data'
data = experiment+'aw_motion/data'
pickled = 'data/deep_look/'
photos = experiment+'photos'
filt = experiment+'filt'

files = os.listdir(data)
for h in range(24):
    filtered = list(filter(lambda nam: int(nam.split('.')[1].split(':')[0]) == h, files))
    l_name = light_h+str(h)+l_nam+'.npy'
    p_name = pickled+str(h)+p_nam+'.pkl'
    if str(h)+p_nam+'.pkl' in os.listdir(pickled):
        print('Hour {} processed.'.format(h))
        continue
    
    for c,i in enumerate(filtered):
        [date, M] = la.process(data+'/'+i); # This should always work if the line before was run
        print(M[1])
        try:
            L = np.load(l_name); L = np.rot90(L); 
        except ValueError:
            pass# This rot is to make the images follow the same pattern
        # This is looking at the raw influence of light on the motion of the frame
        light_red = gaussian_filter(np.rot90(L, k=3).astype(int), sigma=7)
        motion_red = np.sum(np.absolute(M), axis=2)
        # # This is trimming our analysis to the good data
        front = 0.3; back = 0.74
        light_red = spp.edge(light_red, (front, back))
        motion_red = spp.edge(motion_red, (front, back))

        # This is looking at how the change in values is related to the other values
        light_change = np.gradient(light_red)
        motion_change = np.gradient(motion_red)

        # All the values that are going into the system
        l = light_red.flatten(); l_cx = light_change[0].flatten(); l_cy = light_change[1].flatten(); m = motion_red.flatten(); m_cx = motion_change[0].flatten(); m_cy = motion_change[1].flatten()

        if c==0:
            data_setup = {'light':l, 'light_dx':l_cx, 'light_dy':l_cy, 'raw_motion':m, 'motion_dx':m_cx, 'motion_dy':m_cy}
            df = pd.DataFrame(data=data_setup)
        else:
            data_setup = {'light':l, 'light_dx':l_cx, 'light_dy':l_cy, 'raw_motion':m, 'motion_dx':m_cx, 'motion_dy':m_cy}
            dapp = pd.DataFrame(data=data_setup)
            df.append(dapp)

        # This next line removes outliers
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    df.to_pickle(p_name)
    print(df)

    # df_corr = df.corr() # This is going to give us a correlation matrix to play with

        #print(df_corr)
