
from quant_mo_memory_profiler import *
import numpy as np
def reshapeSave(file_s):
    tempArr = np.array([], dtype=int)
    if diff:
        # This will need to be modified in the future
        assert len(file_s) >= 2
        [a_date, a_arr] = np.load(files[0])
        [b_date, b_arr] = np.load(files[1])
        u_array = b_arr - a_arr
    else:
        for i in file_s:
            temp = process(i)
            if temp:
                [temp_date, arr] = temp
                dates.append(temp_date)
                if sys.getsizeof(tempArr) == 96:
                    tempArr = reshapeHelp(arr)
                else:
                    tempIn = reshapeHelp(arr)
                    tempArr = np.concatenate((tempArr, tempIn), axis=0)

                #print('Shape of the array is: {}'.format(tempArr.shape))
                #print('Size of the array is: {}\n'.format(sizeof_fmt(sys.getsizeof(tempArr))))

        u_array = np.average(tempArr, axis=0)

    print('Size of the final average: {}\n'.format(sizeof_fmt(sys.getsizeof(u_array))))
    print(u_array.shape)

    saveTxt(u_array)