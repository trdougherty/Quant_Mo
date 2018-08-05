from memory_profiler import profile
from uncertainties import unumpy
import uncertainties as u

@profile
def uncert(arr):
    assert type(arr).__module__ == np.__name__ # This confirms we're working with a numpy array
    assert len(arr.shape) >= 3 # This confirms we're working with at least a 4D array
    a = unumpy.uarray(np.sum(arr,axis=0)/arr.shape[0],np.std(arr,axis=0))
    return a