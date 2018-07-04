from __future__ import print_function
# Better samples/python2/opt_flow.py
# for Raspberry Pi

## reference
# - http://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/
# - http://stackoverflow.com/questions/2601194/displaying-a-webcam-feed-using-opencv-and-python
# - http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from OpticalFlowShowcase import *
import numpy as np
#from imutils.video import VideoStream
#import imutils
import argparse
import io
import signal
import os
import sys

usage_text = '''
Pass -t to specifiy time of recording at command line

Pass
Hit 's' to save image.

Hit ESC to exit.
'''
# Sets video writing variable for future use
writer = None
camera = None

def signal_handler(signal, frame):
    print('System is gracefully exiting..')
    if writer is not None:
        writer.release()
        print('Closed video writing system')
    if camera is not None:
        camera.close()
        print('Closed camera')
    print('Closing file....')
    sys.exit(0)

# These prep the system for interrupts
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGALRM, signal_handler)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
ap.add_argument("-p", "--picamera", type=int, default=1,
	help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-f", "--fps", type=int, default=32,
	help="FPS of output video")
ap.add_argument("-t", "--time", type=int, default=30,
	help="Duration of output video. Default is 30s.")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
	help="codec of output video")
args = vars(ap.parse_args())

# initialize the video stream and allow the camera
print("[INFO] warming up camera...")

res_x = 64
res_y = 64
camera = PiCamera()
camera.resolution = (res_x, res_y)
camera.framerate = 60
rawCapture = PiRGBArray(camera, size=(res_x, res_y))

# Warmup time for the camera
time.sleep(0.1)
lastTime = time.time()*100.0

# initialize the FourCC, video writer, dimensions of the frame, and
# zeros array
fourcc = cv2.VideoWriter_fourcc(*args["codec"])

#These will be used for developing the motion vectors required to capture user motion
(h, w) = (None, None)
zeros = None

def main():
    # initialize the FourCC, video writer, dimensions of the frame, and
    # zeros array
    t0 = time.time()
    fourcc = cv2.VideoWriter_fourcc(*args["codec"])
    writer = None

    #looping value
    loop = 80
    count = 0

    #Builds the appropriately timed interval
    end_time = time.time() + int(args["time"])

    ## main work
    print("Setup: \t{}".format(time.time()-t0))
    t_fps = time.time()

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # get array & clear the stream in preparation for the next frame
        t1 = time.time()
        if(t1>end_time):
            break
        image = frame.array
        rawCapture.truncate(0)

        print("Capture: \t{}".format(time.time()-t1))

        #adds the counter
        count += 1

	#Builds the method to write the frames into video
        if writer is None:
            print(image.shape[:2])
            (h, w) = image.shape[:2]
            writer = cv2.VideoWriter(args["output"], fourcc, args["fps"], (w, h), True)
            #zeros = np.zeros((h, w), dtype="uint8")

        t2 = time.time()
	#Output matrix
        output = np.zeros((h, w, 3), dtype="uint8")
        output[0:h, 0:w] = image
        writer.write(output)

        print("Writing: \t{}".format(time.time()-t2))
        key = cv2.waitKey(1)
        print(" ")

    ## finish
    camera.close()
    writer.release()
    print("Effective FPS was: \t{}".format(count/(time.time()-t_fps)))

if __name__ == '__main__':
    print(usage_text)
    main()
