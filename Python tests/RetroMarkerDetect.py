import numpy as np
import cv2
import math

#	#Open the first video device and bind to 'cam'
#	cam = cv2.VideoCapture(0)

colorLower = (36, 25, 25)
colorUpper = (70, 255,255)

def alignment_detect(sourceImg):
	#Convert to HSV color space for simpler matching
	hsvImg = cv2.cvtColor(sourceImg, cv2.COLOR_BGR2HSV)
	#Find everything that is 'green'
	maskImg = cv2.inRange(hsvImg, colorLower, colorUpper)

	# NB: using _ as the variable name for two of the outputs, as they're not used
	_, contours, _ = cv2.findContours(maskImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	print ("Contours = ", contours)
	blob = max(contours, key=lambda el: cv2.contourArea(el))
	M = cv2.moments(blob)
	center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
	print (center)
	cv2.circle(sourceImg, center, 2, (0,0,255), -1)

	except (ValueError, ZeroDivisionError):
		pass

#	params = cv2.SimpleBlobDetector_Params()
#
#	params.filterByColor = True
#	params.blobColor = 0
#
#	blobDetector = cv2.SimpleBlobDetector_create (params)
#	keyPoints = blobDetector.detect(maskImg)
#
#	for keyPoint in keyPoints:
#		x = keyPoint.pt[0]
#		y = keyPoint.pt[1]
#		s = keyPoint.size
#
#		print (x, y, s)
#	im_with_keypoints = cv2.drawKeypoints(maskImg, keyPoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	#Convert mask to gray scale
#	maskGrayImg = cv2.cvtColor(maskImg,cv2.COLOR_BGR2GRAY)

	#Find the contours of each 'blob'
	#im2, contours, hierarchy = cv2.findContours(maskGrayImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	cv2.imshow("sourceImg",          sourceImg)
	cv2.imshow("maskImg",            maskImg)
#	cv2.imshow("im_with_keypoints",            im_with_keypoints)

def draw_cross(img, x, y, color=[255, 0, 0], thickness=3):
	height, width, channels = img.shape 
	# Make a copy of the original image.
	img_out = np.copy(img)
	cv2.line(img_out, (0, y), (width, y), color, thickness)
	cv2.line(img_out, (x, 0), (x, height), color, thickness)
	return img_out


def do_processing():
	global img

	#Load an image from a file into 'img' array
	img = cv2.imread("Retro1.jpg");

	while True:
#		#Capture an image from the camera and assign to 'img' array
#		ret_val, img = cam.read()


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


