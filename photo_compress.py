import sys
import numpy
import cv2

def normalized(a, axis=-1, order=2):
    l2 = numpy.atleast_1d(numpy.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / numpy.expand_dims(l2, axis)

def main():
    thing = sys.argv[1:] #Collects photos to be parsed
    im1 = cv2.imread(thing[0])
    print(im1.shape)
    summative = numpy.empty(im1.shape) #This starts our desired numpy array
    for i in thing:
        im = cv2.imread(i)
        summative = numpy.add(summative,im)
    print("The summative numpy array is: ")
    print(summative)
    print("The normalized numpy array is: ")
    prepped = normalized(summative,2)*255
    final = prepped.astype(int)
    cv2.imwrite("final.jpg",final)

if __name__ == '__main__':
    main()
