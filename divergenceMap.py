import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

x = np.linspace(0,64)
y = np.linspace(0,64)

xx,yy = np.meshgrid(x,y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(xx, yy, yy+xx, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.show()
