import numpy as np
import os
import supplement as sp
import pandas as pd
import seaborn as sns
from tqdm import tqdm

experiment = '/Users/TRD/Research_Personal/Light-Barometer/data/elles/'

## This is the experimental setup for the logistics
data = experiment + 'data'
pickled = experiment + 'pickled'
photos = experiment + 'photos'
filt = experiment + 'filt'

if os.path.exists(pickled) is False:
    os.makedirs(pickled)
if os.path.exists(photos) is False:
    os.makedirs(photos)
if os.path.exists(filt) is False:
    os.makedirs(filt)

## This looks at the collected data and loads it
files = os.listdir(data)
full_files = [ data+'/'+i for i in files ]
A = np.load(full_files[0], encoding='latin1')

## Sets the datetime values in the pandas array for filtering later
tee = [ i.strip('.npy').split('@',1)[-1] for i in files ]
light = pd.DataFrame({'filename': files,'date': tee})
light['date'] = pd.to_datetime(light['date'], format='%Y-%m-%d.%H:%M:%S')
light['hour'] = pd.DatetimeIndex(light['date']).hour
dis_hours = sorted(light['hour'].drop_duplicates().values)

my_min = my_max = 0
max_hour = min_hour = 0
for HOUR in dis_hours:
    outfile = str(HOUR)
    e_name = 'aw_light'
    filtered = light[light['hour']==HOUR]['filename'].values
    f_name = filt+'/'+outfile+e_name+'.npy'
    # This loads the file if need
    if os.path.isfile(f_name):
        M = np.load(f_name)
    # This builds a new file if need
    else:
        tempArr = np.array([], dtype='float16')
        for i in tqdm(filtered):
            [date, heat_arr, light_arr] = sp.process(i, data)
            if light_arr is not None:
                if tempArr.size == 0:
                    tempArr = sp.reshapeHelp(light_arr)
                else:
                    tempArr = np.concatenate([tempArr, sp.reshapeHelp(light_arr)], axis=0)
        
        M = np.mean(tempArr, axis=0)
        np.save(filt+'/'+outfile+e_name, M)
        
# Going to be looking for max and mins between the arrays
'''

'''
for HOUR in range(HOURS):
    outfile = str(HOUR)
    if HOUR == 0:
        a_file = str(HOUR+1)
        b_file = '0'
    elif HOUR == HOURS-1:
        a_file = str(HOURS-1)
        b_file = str(HOUR-1)
    else:
        a_file = str(HOUR+1)
        b_file = str(HOUR-1)
    
    f_name = filt+'/'+outfile+e_name+'.npy';
    b_name = filt+'/'+b_file+e_name+'.npy';
    a_name = filt+'/'+a_file+e_name+'.npy';
    N = np.load(f_name); 
    N_a = np.load(a_name); 
    N_b = np.load(b_name)
    if np.amin(N-N_a) < my_min or np.amin(N-N_b) < my_min:
        my_min = np.amin(M)
        min_hour = HOUR
    if np.amax(N-N_a) > my_max or np.amax(N-N_b) > my_max:
        my_max = np.amax(M)
        max_hour = HOUR
        '''
            
print("Completed process")
