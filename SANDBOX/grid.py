from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

def distance(point, x, y):
    return np.power(np.power(point[0]-x,2)+np.power(point[1]-y,2),2)

x = np.linspace(0, 64, num=64)
y = np.linspace(0, 64, num=64)
point = [30, 20]

X, Y = np.meshgrid(x, y)
Z = distance(point, X, Y)

def dim(x):
    return x.shape

print(dim(X))
print(dim(Y))
print(dim(Z))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');

plt.show()
