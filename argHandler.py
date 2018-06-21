import sys
import numpy as np

imported = sys.stdin.read()
files = imported.rstrip().split('\n')
print(files)

def init(file):
    first_file = np.load(file)
    print(first_file)

init(files[0])
