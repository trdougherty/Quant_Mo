import pandas as pd

t = pd.read_pickle('data/aw_motion/pickled/full_motion.pkl')
print(t.shape[0])
print('Number of frames processed is: {}'.format(t.shape[0]/(64*64*2)))