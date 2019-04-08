#This code looks for green retro-flective markers in an image
#The 2 markers nearest the center line are then averaged to 
#determine the 'center' of the 2 marker guide. This can then 
#be used to guide a mechanism in both X and Y
#The ouptut result is in terms of -1 to +1 in each of X & Y.

import numpy as np
import cv2
import math

#Open the first video device and bind to 'cam'
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 352)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)

#First parameter is basically the color range we are looking for, i.e. green
#Second parameter is basicaaly how much of the color exists (saturation)
#Third parameter is basically the brightness/contrast of the color
#colorLower = (36, 25, 25)
#colorUpper = (70, 255,255)
#colorLower = (32, 50, 100)
#colorUpper = (85, 255,255)
#colorLower = (56, 25, 100)
#colorUpper = (70, 255,255)
colorLower = (32, 45, 25)
colorUpper = (75, 255,255)

ga = np.uint8([[[10/2, 255/2, 236/2]]])
gb = np.uint8([[[215/2, 255/2, 8/2]]])
gc = np.uint8([[[184/2, 255/2, 236/2]]])
gd = np.uint8([[[246/2, 255/2, 197/2]]])

hsv_ga = cv2.cvtColor(ga,cv2.COLOR_BGR2HSV)
hsv_gb = cv2.cvtColor(gb,cv2.COLOR_BGR2HSV)
hsv_gc = cv2.cvtColor(gc,cv2.COLOR_BGR2HSV)
hsv_gd = cv2.cvtColor(gd,cv2.COLOR_BGR2HSV)

#print "a = ", hsv_ga
#print "b = ", hsv_gb
#print "c = ", hsv_gc
#print "d = ", hsv_gd

def alignment_detect(sourceImg):
	#Find the image size and center
	width, height = height, width = sourceImg.shape[:2]
	midX = int(width / 2)
	midY = int(height / 2)
	#Convert to HSV color space for simpler matching
	hsvImg = cv2.cvtColor(sourceImg, cv2.COLOR_BGR2HSV)
	blurImg = cv2.GaussianBlur(hsvImg, (3, 3), 0)
	#Find everything that is 'green'
	inRangeImg = cv2.inRange(blurImg, colorLower, colorUpper)
	#'erode' small features to get rid of noise
	erodeImg = cv2.erode(inRangeImg, None, iterations=5)
	#'dilate' back
	maskImg = cv2.dilate(erodeImg, None, iterations=10)

	#Find the contours of the 'blobs'
	contours, hierarchy = cv2.findContours(maskImg.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	center = None
	#Run through contours and draw circles, adding coordinates and size to a list
	location, displayImg = findTargets(contours, sourceImg)
	#Convert from -1 +1 to window sized/centerd coordinates
	locationWindow = (int(((location[0] * width) + width) / 2) , int(((location[1] * height) + height) / 2))
	#Draw center line guide
	displayImg = draw_cross(displayImg, (midX, midY), color=[255, 0, 0], thickness=1)
	#Draw target centers guide
	displayImg = draw_cross(displayImg, locationWindow, color=[140, 0, 200], thickness=2)

	cv2.imshow("sourceImg",   sourceImg)
	cv2.imshow("maskImg",     maskImg)
	cv2.imshow("inRangeImg",  inRangeImg)
	cv2.imshow("displayImg",  displayImg)
	cv2.imshow("erodeImg",  erodeImg)

def findTargets(contours, imgIn):
	#Find discrete blobs in the list of contours provided
	#Points are converted to -1 to +1 so not dependent on source image size
	blobLocations = []
	imgOut = np.copy(imgIn)
	#Find the image size and center
	width, height = height, width = imgIn.shape[:2]
	midX = int(width / 2)
	midY = int(height / 2)
	#Only proceed if see 2 or more blobs
	loop = 0
	for marker in contours:
		loop = loop + 1
		#Find the enclosing circle coordinates & size
		((x, y), radius) = cv2.minEnclosingCircle(marker)
		#Find the 'moment' of the pixels in the blob
		M = cv2.moments(marker)
		#Make sure they are valid. If not default to the middle of the image, i.e. straight ahead
		if (M["m00"] != 0):
			(cx, cy) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		else:
			(cx, cy) = (midX, midY)
		#Check if the blob has some reasonable size (filter small noise blobs)
		if radius > 20:
			# draw the centroid on the frame,
			cv2.circle(imgOut, (cx, cy), 5, (0, 0, 255), -1)
			cv2.circle(imgOut, (cx, cy), int(radius), (0, 255, 255), 2)
			cxn = float(cx - midX) / midX
			cyn = float(cy - midY) / midY
			blobLocations.append((cxn, cyn))
	#Now cycle through all blobs to find 2 closest to the centerline then find center of those 2
	targetCenter = findCenterMid(blobLocations)
	return targetCenter, imgOut

def findCenterMid(points):
	print "Source points = ", points
	if (len(points) < 2):#Less than 2 points so can't really track markers. Keep going straight
		return (0, 0)
	elif (len(points) == 2): #Only 2 points found so just use them directly
		return findCenter(points[0], points[1])
	else:
		#Multiple points so need to find the closest 2 points to the center line
		entries = len(points)
		p1 = (10000, 10000)
		p2 = (10000, 10000)
		for loop in range(0, entries):
			p1, p2 = findSmallestTwo(points[loop], p1, p2)

	return findCenter(p1, p2)

def findSmallestTwo(p1, p2, p3):
	#Check the x deviations from center and pick the smallest 2 from 3
	print "in = " , p1, p2, p3
	if (abs(p1[0]) < abs(p2[0])):         #p1 < p2, check where p3 is
		if (abs(p2[0]) < abs(p3[0])):
			(o1, o2) = (p1, p2)
		else :
			(o1, o2) = (p1, p3)
	else:                                 #p2 < p1, check where p3 is
		if (abs(p3[0]) < abs(p1[0])):
			(o1, o1) = (p2, p3)
		else :
			(o1, o2) = (p2, p1)
	print "out = " , o1, o2
	return o1, o2

def findCenter(p1, p2):
	return ((p1[0] + p2[0]) / 2),((p1[1] + p2[1]) / 2) 

def draw_cross(img, (x, y), color=[255, 0, 0], thickness=3):
	height, width, channels = img.shape 
	# Make a copy of the original image.
	img_out = np.copy(img)
	cv2.line(img_out, (0, y), (width, y), color, thickness)
	cv2.line(img_out, (x, 0), (x, height), color, thickness)
	return img_out

def do_processing():
	global img

	#Load an image from a file into 'img' array
	img = cv2.imread("Retro2.jpg");

	while True:
		#Capture an image from the camera and assign to 'img' array
#		ret_val, img = cap.read()


		#Pass the image in to the alignment detector
		print
		print
		print "Processing new image"
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


