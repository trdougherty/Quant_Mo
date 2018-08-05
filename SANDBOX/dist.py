from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

start = 0; end = 5

x = np.arange(start,end,0.01)
y = x

xx, yy = np.meshgrid(x,y)

def distance(point, x, y):
    return np.power(np.power(point[0]-x,2)+np.power(point[1]-y,2),2)

Z = distance([1,2],xx,yy)

plt.figure()
cp = plt.contourf(xx, yy, Z)
plt.colorbar(cp)
plt.title('Filled Contours Plot')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.show()
