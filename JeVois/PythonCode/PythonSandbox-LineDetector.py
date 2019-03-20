import libjevois as jevois
import cv2
import numpy as np

## Simple example of image processing using OpenCV in Python on JeVois
#
# This module is here for you to experiment with Python OpenCV on JeVois.
#
# By default, we get the next video frame from the camera as an OpenCV BGR (color) image named 'inimg'.
# We then apply some image processing to it to create an output BGR image named 'outimg'.
# We finally add some text drawings to outimg and send it to host over USB.
#
# See http://jevois.org/tutorials for tutorials on getting started with programming JeVois in Python without having
# to install any development software on your host computer.
#
# @author Laurent Itti
# 
# @videomapping YUYV 352 288 30.0 YUYV 352 288 30.0 JeVois PythonSandbox
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2017 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class PythonSandbox:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("sandbox", 100, jevois.LOG_INFO)
        
    # ###################################################################################################

    ## Process function with USB output
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR by default. If
        # you need a grayscale image instead, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB()
        # and getCvRGBA():
        inimg = inframe.getCvBGR()
        #inimg = inframe.getCvGRAY()
        
        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

        # Detect edges using the Laplacian algorithm from OpenCV:
        #
        # Replace the line below by your own code! See for example
        # - http://docs.opencv.org/trunk/d4/d13/tutorial_py_filtering.html
        # - http://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
        # - http://docs.opencv.org/trunk/d5/d0f/tutorial_py_gradients.html
        # - http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
        #
        # and so on. When they do "img = cv2.imread('name.jpg', 0)" in these tutorials, the last 0 means they want a
        # gray image, so you should use getCvGRAY() above in these cases. When they do not specify a final 0 in imread()
        # then usually they assume color and you should use getCvBGR() here.
        #
        # The simplest you could try is:
        #    outimg = inimg
        # which will make a simple copy of the input image to output.
        #outimg = cv2.Laplacian(inimg, -1, ksize=5, scale=0.25, delta=127)
        #Find the image size
        height, width, channels = inimg.shape
        #Convert to gray scale
        procimgGray = cv2.cvtColor(inimg, cv2.COLOR_RGB2GRAY)
        #Blur it
        procimgBlur = cv2.GaussianBlur(procimgGray, (5, 5), 0)
        #Detect edges
        procimgCanny = cv2.Canny(procimgBlur, 50, 255)
        #Detect Hough lines
        plines_gaus = cv2.HoughLinesP( procimgCanny, rho=6, theta= np.pi / 60, threshold=120, lines=np.array([]), minLineLength=5, maxLineGap=21)
        # Create a blank 3 channel image that matches the original in size.
        imgPlines = np.zeros( ( height, width, 3), dtype=np.uint8,)
        #Draw lines an image
        imgPlines = self.draw_lines(imgPlines, plines_gaus, (0, 0,255), 3)
        
        #At this point we have the following images available...
        #inimg        = original image             - 3 channel color
        #procimgGray  = gray scale of input image  - 1 channel color
        #procimgBlur  = Gausian blured image       - 1 channel color
        #procimgCanny = Canny edges image          - 1 channel color
        #imgPlines    = Hough detected lines       - 3 channel color
        #
        
#        #Convert processed image back to BGR so we can merge with original BGR image
        procimagebgr = cv2.merge((procimgCanny, procimgCanny, procimgCanny))
#        #Merge desired intermediate images
        outimg = cv2.add(imgPlines, procimagebgr) 
        
        #Draw target guide lines to aid driver
        cv2.line(outimg, ((int)(width / 2), 0),        ((int)(width / 2), height), (255, 0, 0), 1)
        cv2.line(outimg, ((int)(width / 2) - 20, 100), ((int)(width / 2) - 40, height), (255, 0, 0), 1)
        cv2.line(outimg, ((int)(width / 2) + 20, 100), ((int)(width / 2) + 40, height), (255, 0, 0), 1)
                
        # Write a title:
        cv2.putText(outimg, "Robocats Vision Guide", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        
        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        #Check output image size
        outheight, outwidth, outchannels = outimg.shape
        #Display some facts at the bottom of the image
        cv2.putText(outimg, fps, (3, outheight - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

        # Convert our OpenCv output image to video output format and send to host over USB:
        outframe.sendCv(outimg)

    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=3):
        # If there are no lines to draw, exit.
        if lines is None:
            return img

        # Loop over all lines and draw them on the blank image.
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

        # Return the lines image.
        return img