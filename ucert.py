from uncertainties import unumpy
import uncertainties as u
import numpy as np

def uncert(arr):
    assert len(arr.shape) >= 3 # This confirms we're working with at least a 4D array
    a = unumpy.uarray(np.sum(arr,axis=0)/arr.shape[0],np.std(arr,axis=0))
    return a