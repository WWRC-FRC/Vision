#Based of multiple sources including the following...
#https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

import numpy as np
import cv2
import math

#lower = (36, 25, 25)
#upper = (70, 255,255)
lower = (29, 86, 6)
upper = (64, 255,255)

def alignment_detect(sourceImg):
        frame = sourceImg

        canvas = frame.copy()
	hsvImg = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
	
        maskImg = cv2.inRange(hsvImg, lower, upper)

        try:
	    print ("In")
            # NB: using _ as the variable name for two of the outputs, as they're not used
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	    print ("Found")
            blob = max(contours, key=lambda el: cv2.contourArea(el))
            M = cv2.moments(blob)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            cv2.circle(canvas, center, 2, (0,0,255), -1)

        except (ValueError, ZeroDivisionError):
            pass

        cv2.imshow('frame',frame)
        cv2.imshow('canvas',canvas)
        cv2.imshow('mask',mask)	

def do_processing():
	global img

	#Load an image from a file into 'img' array
	img = cv2.imread("Retro1.jpg");

	while True:


		#Pass the image in to the alignment detector
		alignment_detect(img)

		if cv2.waitKey(0) == 27: 
			break  # esc to quit

def main():
#	init()
	do_processing()
	#Cleanup the display windows
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()


