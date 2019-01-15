import numpy as np
from uncertainties import unumpy
import pandas as pd
from tqdm import tqdm
import os

def edge(arr, scope):
    if len(scope) == 2:
        lowerBound_x = lowerBound_y = scope[0]
        upperBound_x = upperBound_y = scope[1]
    elif len(scope) == 4:
        lowerBound_x = scope[0]; lowerBound_y = scope[2]
        upperBound_x = scope[1]; upperBound_y = scope[3]
    else: return None
    x_ = arr.shape[0]; y_ = arr.shape[1]
    return arr[int(x_*lowerBound_x):int(x_*upperBound_x),int(y_*lowerBound_y):int(y_*upperBound_y),:]

def pythag(arr):
    # This is going to reduce the system into one array ~ ideally ~ a mxm matrix of rank 2
    finalArr = np.zeros(arr.shape[:-1],dtype='float32')
    for i in range(arr.shape[-1]):
        finalArr += np.power(arr[...,i], 2)
    return np.power(finalArr, 0.5)

def process(file, path_name):
    try:
        numObj = np.load(path_name+'/'+file, encoding='latin1')
        try:
            [date, arr] = numObj
        except ValueError:
            return numObj
        A = unumpy.matrix(arr.flatten())
        A_nom = np.ravel(A.nominal_values)
        outArr = np.reshape(A_nom, arr.shape)
        return [date, outArr]
    except EOFError:
        return None

def gradient(array): return np.gradient(array) #np.add.reduce(np.gradient(array))

def importMatrix(file):
    num_cols = 2
    converters = dict.fromkeys(
        range(num_cols),
        lambda col_bytes: u.ufloat_fromstr(col_bytes.decode("latin1")))
    arr = np.loadtxt(file, converters=converters, dtype=object)
    return arr.reshape((64,64,2))

# Shouldn't really be needed
def normalize(x):
    return x/np.amax(np.absolute(x))

def reshapeHelp(arr):
    # This function converts the array into an added dimension to support concatenation later
    assert type(arr).__module__ == np.__name__
    lis = list(arr.shape)
    lis.insert(0, 1)
    return np.reshape(arr, tuple(lis)).astype('float16')

def floaty(*args):
    ret = []
    for i in args:
        if type(i) is np.ndarray: 
            ret.append(i.astype('float64'))
        else:
            raise ValueError
    return ret

def set_paths(experiment, e_name='aw_motion'):
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

def chunk_hours(inp, outp, e_name='aw_motion'):
    set_paths(inp, e_name)
    motion_files = [f for f in os.listdir(inp) if os.path.isfile(os.path.join(inp, f))]
    HOURS = 24
    my_min = my_max = 0
    max_hour = min_hour = 0
    for HOUR in range(HOURS):
        # This is doing all the formatting we need
        outfile = str(HOUR)
        tee = [ i.strip('.npy').split('@',1)[-1] for i in motion_files ]
        dates = pd.DataFrame({'filename': motion_files,'date': tee})
        dates['date'] = pd.to_datetime(dates['date'], format='%Y-%m-%d.%H:%M:%S')
        dates['hour'] = pd.DatetimeIndex(dates['date']).hour
        filtered = dates[dates['hour']==HOUR]['filename'].values
        f_name = outp+'/'+outfile+e_name+'.npy'
        print(f_name)
        # This loads the file if need
        if os.path.isfile(f_name):
            M = np.load(f_name)
        # This builds a new file if need
        else:
            tempArr = np.array([], dtype='float16')
            for i in tqdm(filtered):
                temp = process(i, inp)
                if temp is not None:
                    [temp_date, arr] = temp
                    if tempArr.size == 0:
                        tempArr = reshapeHelp(arr)
                    else:
                        tempArr = np.concatenate([tempArr, reshapeHelp(arr)], axis=0)
            M = np.mean(tempArr, axis=0)
            np.save(outp+'/'+outfile+e_name, M)
        
    # Going to be looking for max and mins between the arrays
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
        
        f_name = outp+'/'+outfile+e_name+'.npy'
        b_name = outp+'/'+b_file+e_name+'.npy'
        a_name = outp+'/'+a_file+e_name+'.npy'
        N = np.load(f_name); 
        N_a = np.load(a_name); 
        N_b = np.load(b_name)
        if np.amin(N-N_a) < my_min or np.amin(N-N_b) < my_min:
            my_min = np.amin(M)
            min_hour = HOUR
        if np.amax(N-N_a) > my_max or np.amax(N-N_b) > my_max:
            my_max = np.amax(M)
            max_hour = HOUR