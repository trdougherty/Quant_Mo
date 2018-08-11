from __future__ import print_function
# Better samples/python2/opt_flow.py
# for Raspberry Pi

## reference
# - http://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/
# - http://stackoverflow.com/questions/2601194/displaying-a-webcam-feed-using-opencv-and-python
# - http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/
# py_video_display.html

import time
import cv2
import numpy as np
#from imutils.video import VideoStream
#import imutils
import argparse
from datetime import datetime
from uncertainties import unumpy
import ucert
import flow

usage_text = '''
Flags:

Pass 'o' for output filename (mandatory).
Pass 'i' for input filename (mandatory).
Pass 'f' for fps (default 17).

Hit ESC to exit.
'''

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
ap.add_argument("-r","--raw_file", required=True,
    help="Provides a path to save the raw file form")
ap.add_argument("-i", "--input", required=True,
	help="path to input video file")
ap.add_argument("-f", "--fps", type=int, default=17,
	help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
	help="codec of output video")
args = vars(ap.parse_args())

# These will be used for developing the motion vectors required to capture user motion
(h, w) = (None, None)
zeros = None

def main():
	#Now we have the appropriate video file retrieval information under cap
    t0 = time.time()
    cap = cv2.VideoCapture(args["input"])
    initialized = False

    # initialize the FourCC, video writer, dimensions of the frame, and
    # zeros array
    fourcc = cv2.VideoWriter_fourcc(*args["codec"])
    writer = None

    while(cap.isOpened()):
        try:
            ret, frame = cap.read()
            if(initialized == False):
            #Builds the appropriate optical flow
                of = flow.Flow(frame) #dense_hsv')
                initialized = True
                ret, frame = cap.read() #Sets the frames again to appropriately build the first set
                iterations = 0
                flowSum = []

            if ret == True:
                flowSum.append(of.calc(frame))
            else:
                break
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            print("Process was Terminated.")

    #if args["polar"]: writer.release()
    cv2.destroyAllWindows()

    t = str(datetime.now())
    X = ucert.uncert(flowSum)

    arrayStorage = np.array([t,X], dtype=object)
    np.save(args["raw_file"], arrayStorage)


if __name__ == '__main__':
    print(usage_text)
    main()
