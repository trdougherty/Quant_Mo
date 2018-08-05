import numpy as np
import pdb
import sys
from memory_profiler import profile
from uncertainties import unumpy
import time

@profile
def build_rand(inter):
    l = []
    for i in range(int(inter)):
        l.append(np.random.randint(0,255,(64,64,2)))
    time.sleep(1)
    return np.array(l,dtype='uint8')

@profile
def convert(arr):
    assert type(arr).__module__ == np.__name__ # This confirms we're working with a numpy array
    assert len(arr.shape) >= 3 # This confirms we're working with at least a 4D array
    a = unumpy.uarray(np.sum(arr,axis=0)/arr.shape[0],np.std(arr,axis=0))
    time.sleep(1)
    return a

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    m = build_rand(sys.argv[len(sys.argv)-1])
    out = convert(m)
    print(out[0])
