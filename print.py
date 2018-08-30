from __future__ import print_function
import time
import cv2
from OpticalFlowShowcase import *
import numpy as np
import argparse
import io
import sys
import datetime
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from memory_profiler import profile
from uncertainties import unumpy
import uncertainties as u
import ucert
import plotly.plotly as py
from plotly.graph_objs import *

# print(resource.getrlimit(resource.RLIMIT_STACK))
# print(sys.getrecursionlimit())

# max_rec = 0x100000

# # May segfault without this line. 0x100 is a guess at the size of each stack frame.
# resource.setrlimit(resource.RLIMIT_STACK, [0x100 * max_rec, resource.RLIM_INFINITY])
# sys.setrecursionlimit(max_rec)

#Variables of interest
## NOTE - 0 is X and 1 is Y
axis = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video file"),
ap.set_defaults(input=False)
args = vars(ap.parse_args())

def process(file):
    return np.load(file)

# Not really needed for this
def localize(point, x, y, mv = 0.08):
    Z = 1/np.power(np.power(point[0]-x,2)+np.power(point[1]-y,2),0.5)
    Z[ Z > mv ] = mv
    return Z*(1/mv)

# This will give use the divergence of the array which we can use for localizing later
def gradient(array): return np.gradient(array) #np.add.reduce(np.gradient(array))

def importMatrix(file):
    num_cols = 2
    converters = dict.fromkeys(
        range(num_cols),
        lambda col_bytes: u.ufloat_fromstr(col_bytes.decode("latin1")))
    arr = np.loadtxt(args["input"], converters=converters, dtype=object)
    return arr.reshape((64,64,2))
    

def printArr(arr, axis):
    # Allows us to work with the shape off the photo we're looking at
    x_dist = np.arange(0,arr.shape[0])
    y_dist = np.arange(0,arr.shape[1])

    x, y = np.meshgrid(x_dist, y_dist)
    z = arr[:,:,axis]

    # min_ = -1; max_ = 1 # This is the default value
    # if (Z.min() < -1): min_ = Z.min()
    # if (Z.max() > 1): max_ = Z.max()

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.contour3D(xx, yy, Z, arr.shape[0], cmap='binary')
    # ax.set_title('{} Axis'.format(axis))
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.set_zlim(min_,max_)
    # plt.show()
    colorscale=[[0.0, 'rgb(20,29,67)'],
        [0.1, 'rgb(28,76,96)'],
        [0.2, 'rgb(16,125,121)'],
        [0.3, 'rgb(92,166,133)'],
        [0.4, 'rgb(182,202,175)'],
        [0.5, 'rgb(253,245,243)'],
        [0.6, 'rgb(230,183,162)'],
        [0.7, 'rgb(211,118,105)'],
        [0.8, 'rgb(174,63,95)'],
        [0.9, 'rgb(116,25,93)'],
        [1.0, 'rgb(51,13,53)']]

    textz=[['x: '+'{:0.5f}'.format(x[i][j])+'<br>y: '+'{:0.5f}'.format(y[i][j])+
        '<br>z: '+'{:0.5f}'.format(z[i][j]) for j in range(z.shape[1])] for i in range(z.shape[0])]

    trace1= go.Surface(z=z,
                x=x,
                y=y,
                colorscale=colorscale,
                text=textz,
                hoverinfo='text',
                )
    
    axis = dict(
    showbackground=True,
    backgroundcolor="rgb(230, 230,230)",
    showgrid=False,
    zeroline=False,
    showline=False)

    ztickvals=list(range(64))
    layout = go.Layout(title="Projections of a surface onto coordinate planes" ,
                    autosize=False,
                    width=700,
                    height=600,
                    scene=dict(xaxis=dict(axis, range=[0, 64]),
                                yaxis=dict(axis, range=[0, 64]),
                                zaxis=dict(axis , tickvals=ztickvals),
                                aspectratio=dict(x=1,
                                                y=1,
                                                z=0.95)
                            )
                    )
    
    z_offset=(np.min(z)-2)*np.ones(z.shape)#
    x_offset=np.min(x_dist)*np.ones(z.shape)
    y_offset=np.min(y_dist)*np.ones(z.shape)

    proj_z=lambda x, y, z: z#projection in the z-direction
    colorsurfz=proj_z(x,y,z)
    proj_x=lambda x, y, z: x
    colorsurfx=proj_z(x,y,z)
    proj_y=lambda x, y, z: y
    colorsurfy=proj_z(x,y,z)

    textx=[['y: '+'{:0.5f}'.format(y[i][j])+'<br>z: '+'{:0.5f}'.format(z[i][j])+
            '<br>x: '+'{:0.5f}'.format(x[i][j]) for j in range(z.shape[1])]  for i in range(z.shape[0])]
    texty=[['x: '+'{:0.5f}'.format(x[i][j])+'<br>z: '+'{:0.5f}'.format(z[i][j]) +
            '<br>y: '+'{:0.5f}'.format(y[i][j]) for j in range(z.shape[1])] for i in range(z.shape[0])]

    tracex = go.Surface(z=z,
                    x=x_offset,
                    y=y,
                    colorscale=colorscale,
                    showlegend=False,
                    showscale=False,
                    surfacecolor=colorsurfx,
                    text=textx,
                    hoverinfo='text'
                )
    tracey = go.Surface(z=z,
                    x=x,
                    y=y_offset,
                    colorscale=colorscale,
                    showlegend=False,
                    showscale=False,
                    surfacecolor=colorsurfy,
                    text=texty,
                    hoverinfo='text'
                )
    tracez = go.Surface(z=z_offset,
                    x=x,
                    y=y,
                    colorscale=colorscale,
                    showlegend=False,
                    showscale=False,
                    surfacecolor=colorsurfx,
                    text=textz,
                    hoverinfo='text'
                )

    data=[trace1, tracex, tracey, tracez]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
    return


# Shouldn't really be needed

def normalize(x):
    return x/np.amax(np.absolute(x))

def intensity(x):
    # This is taking the hypotenuse of the intensity vectors
    assert type(x).__module__ == np.__name__
    return np.power(np.sum(np.power(x,2),axis=len(x.shape)-1),0.5) # pythagorean


if __name__ == '__main__':
    file = args["input"]
    A = importMatrix(file)
    A_U = unumpy.uarray(*A)