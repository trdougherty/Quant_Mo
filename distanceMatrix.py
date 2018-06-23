from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import math

start = 0; end = 10
max_val = 0.08

x = np.arange(start,end,0.1)
y = x

xx, yy = np.meshgrid(x,y)

def distance(point, x, y):
    return 1/np.power(np.power(point[0]-x,2)+np.power(point[1]-y,2),0.5)

def dist_power(x):
    return 1+(-1/(1+math.e**(-10*x/90.5 - 4)))

def divergence(F):
    """ compute the divergence of n-D scalar field `F` """
    return np.add.reduce(np.gradient(F))

Z = distance([15,20],xx,yy)
#Z = 1/(xx+yy)
Z[ Z > max_val ] = max_val
Z = Z*(1/max_val)
print(Z)
Z = divergence(Z)
print(Z)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(xx, yy, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.show()

#plt.colored(cp)
#plt.title('Filled Contours Plot')
#plt.xlabel('x (cm)')
#plt.ylabel('y (cm)')
#plt.show()
