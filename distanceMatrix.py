from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import math

start = 0; end = 64
max_val = 0.3

x = np.arange(start,end)
y = x

xx, yy = np.meshgrid(x,y)

def distance(point, x, y):
    return 1/np.power(np.power(point[0]-x,2)+np.power(point[1]-y,2),0.5)

def dist_power(x):
    return 1+(-1/(1+math.e**(-10*x/90.5 - 4)))

def divergenced(field):
    "return the divergence of a n-D field"
    return np.sum(np.gradient(field),axis=0)

def divergence(F):
    """ compute the divergence of n-D scalar field `F` """
    return np.add.reduce(np.gradient(F))

#Z = distance([2,4],xx,yy)
Z = distance([10,15], xx, yy)
print("Original Z:\n{}\n\n".format(Z))
Z[ Z > max_val ] = max_val
Z = Z*(1/max_val)
ZL = divergence(Z)
ZP = divergenced(Z)
print("New Z:\n{}\n\n".format(ZP))
print("Original Dot prod method:\n{}\n\n".format(ZP))
print(ZP.shape)
print(ZL.shape)

print("Dot and Addred are the same: {}".format(ZL==ZP))
#Q = np.gradient(Z)
#print(Q)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(xx, yy, ZL, 50, cmap='binary')
ax.view_init(elev=27., azim=-44)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.show()

#plt.colored(cp)
#plt.title('Filled Contours Plot')
#plt.xlabel('x (cm)')
#plt.ylabel('y (cm)')
#plt.show()
