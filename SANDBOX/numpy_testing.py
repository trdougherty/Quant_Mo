import numpy as np
import pdb
import sys
from memory_profiler import profile
from uncertainties import unumpy

@profile
def build_rand(inter):
    l = []
    for i in range(int(inter)):
        l.append(np.random.randint(0,255,(64,64,2)))
    return np.array(l,dtype='uint8')

@profile
def straw(array):
    return array

if __name__ == "__main__":
    m = build_rand(sys.argv[len(sys.argv)-1])
    print(m[:,1,1,1].shape)
