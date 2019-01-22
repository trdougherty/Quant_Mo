# from __future__ import print_function
import numpy as np
import pandas as pd
import sys

## This is specific to this machine and experiment, should be replaced with config file if published
experiment = 'data/aw_motion/'
light_e = 'data/elles/data/'
light_h = 'data/elles/filt/'
e_name = 'aw_motion'
l_nam = 'aw_light'
data = experiment + 'data'
pickled = experiment + 'pickled'
photos = experiment + 'photos'
filt = experiment + 'filt'

# Hour of interest here:
if len(sys.argv) > 2:
    hour = sys.argv[2]
else:
    raise
h = str(hour)
f_name = filt+'/'+h+e_name+'.npy'
l_name = light_h+h+l_nam+'.npy'
M = np.load(f_name); # This should always work if the line before was run
L = np.load(l_name); L = np.rot90(L); # This rot is to make the images follow the same pattern

# This is looking at the raw influence of light on the motion of the frame
data_setup = {'light':L.flatten(), 'motion':np.sum(np.absolute(M), axis=2).flatten()}
df = pd.DataFrame(data=data_setup)
df_corr = df.corr() # This is going to give us a correlation matrix to play with
if len(sys.argv) > 1:
    if str(sys.argv[1]) == '0':
        print(df.light.to_string(index=False))
    else:
        print(df.motion.to_string(index=False))
else:
    pass

#print(df_corr)
