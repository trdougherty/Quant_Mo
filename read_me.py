import numpy as np

f = open("raw_motion.npy","r")
np.lib.npyio.format.read_magic(f)
print(info)
f.close()
