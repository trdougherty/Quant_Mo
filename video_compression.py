from __future__ import print_function
# Better samples/python2/opt_flow.py
# for Raspberry Pi

## reference
# - http://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/
# - http://stackoverflow.com/questions/2601194/displaying-a-webcam-feed-using-opencv-and-python
# - http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

import time
import cv2
from OpticalFlowShowcase import *
import numpy as np
#from imutils.video import VideoStream
#import imutils
import argparse
import io

usage_text = '''
[INFO] Now converting motion vector video into summative photo...
'''

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
ap.add_argument("-i", "--input", required=True,
        help="path to input file")
args = vars(ap.parse_args())

def main():
    t0 = time.time()
    cap = cv2.VideoCapture(args["input"])
    initialized = False
    counter = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        try:
            if(initialized == False):
                cumulative = np.zeros(frame.shape)
                initialized = True
            #print(np.var(frame)) #This is just a very general idea of how much motion activity is registered
            image = frame
            cumulative = cumulative + image
            counter += 1
        except:
            break

    #Cululative Value recording
#    cumulative = np.zeros((res_y,res_x,3), dtype = np.int32)
    #adds the counter
    print("Took {} seconds to process".format(time.time()-t0))
    max_val = np.amax(cumulative) #This yeilds max value of the array for normalization
    cumulative = cumulative/max_val #Normalizes the PiRGBArray
    final_image = cumulative*255 #Stretches array into color scheme
    cv2.imwrite(args["output"],final_image.astype(int))
    #Builds the appropriate method to record motion. Change this to utilize different motion detection algorithms
        #Saves the image to the recorded vided object

    ## finish
    #writer.release()
    #cv2.imwrite(cumulative/loop)

if __name__ == '__main__':
    print(usage_text)
    main()
