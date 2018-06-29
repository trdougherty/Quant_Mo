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
from OpticalFlowShowcase import *
import numpy as np
#from imutils.video import VideoStream
#import imutils
import argparse
from datetime import datetime

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
ap.add_argument("-p", "--polar", help="Use this to store a visual cue of motion",
                    action="store_true")
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
                of = CreateOpticalFlow('dense_hsv') #dense_hsv')
                of.set1stFrame(frame)
                initialized = True
                ret, frame = cap.read() #Sets the frames again to appropriately build the first set
                iterations = 0
                flowSum = np.zeros([frame.shape[0],frame.shape[1],2]) # this would need to be MANUALLY set for another size of flow vectors

            #Builds the appropriate writing print_function, given that we want to write polar
            if writer is None and args["polar"]:
                (h, w) = frame.shape[:2]
                writer = cv2.VideoWriter(args["output"], fourcc, args["fps"], (w, h), True)

            if ret == True:
                #Sets the time for Analysis. This includes graying the image,
                # building empty motion array, and applying motion vector analysis.
                t2 = time.time()
                #Builds the timing interval for the retrieval
                motion_img = of.apply(frame)
                #cv2.putText(motion_img,"Hello World!!!", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                #x = [motion_img[x,:,:].sum() for x in range(motion_img.shape[0])]
                #print(x)
                #output[0:h, 0:w] = motion_img

                print("Analysis:\t\t{} seconds".format(time.time()-t2))

                t3 = time.time()
                #Builds a summed array of the color values for variance analysis in any direction
                summed_colors = np.sum(motion_img,axis=2)
                #Shows the image and computes the variance of the image
                variance = np.var(summed_colors)
                print("Variance of the matrix:\t{}".format(variance))
                #cv2.imshow('image',motion_img)
                if variance > 6000:
                    if args["polar"]: writer.write(motion_img)
                    iterations += 1
                    flowSum += of.getFlow() # This creates our summed numpy array
                    #for i in range(of.flow.shape[2]):
                    #np.savetxt(args["raw_file"],of.flow[:,:,i])

                    #h5file = tables.open_file(args["raw_file"], "a", driver="H5FD_CORE") #Not sure if this works...
                print("Photo storage:\t\t{} seconds".format(time.time()-t3))
                print(" ") #For spacing
            else:
                break
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            print("Process was Terminated.")

    #if args["polar"]: writer.release()
    cv2.destroyAllWindows()
    print("Total time:\t\t{} seconds".format(time.time()-t0))

    flowShape = flowSum.shape[:2] #Gets the size of the array that we need
    flowSum = flowSum.astype(np.float16)
    t = datetime.now().timestamp()

    arrayStorage = np.array([t,iterations,flowSum[:,:,0],flowSum[:,:,1]], dtype = object)
    np.save(args["raw_file"], arrayStorage)


if __name__ == '__main__':
    print(usage_text)
    main()
