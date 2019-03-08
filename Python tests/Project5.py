#Developed from code at https://medium.com/@mrhwick/simple-lane-detection-with-opencv-bfeb6ae54ec0
#Another useful resource with good function overview https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
import numpy as np
import cv2
import math

CannyLow   = 180
CannyHigh  = 255
Blur       = 9
BlurSD     = 40
BiSigS     = 4
BiSigC     = 153
BiD        = 13
HRho       = 2
HTheta     = 4
HThresh    = 28
HLen       = 10
HGap       = 21

#	#Open the first video device and bind to 'cam'
#	cam = cv2.VideoCapture(0)

def EmptyCallback(var):
	pass

def UpdateParameters():
	global CannyLow
	global CannyHigh
	global Blur
	global BlurSD
	global BiD
	global BiSigC
	global BiSigS
	global HRho
	global HTheta
	global HThresh
	global HLen
	global HGap

	CannyLow  = cv2.getTrackbarPos('Canny Low',     "Controls")
	CannyHigh = cv2.getTrackbarPos('Canny High',    "Controls")
	Blur      = cv2.getTrackbarPos('Gauss Blur',    "Controls") | 1
	BlurSD    = cv2.getTrackbarPos('Gauss Blur SD', "Controls")
	BiD       = cv2.getTrackbarPos('Bi D',          "Controls")
	BiSigC    = cv2.getTrackbarPos('Bi SigC',       "Controls")
	BiSigS    = cv2.getTrackbarPos('Bi SigS',       "Controls")
	HRho      = cv2.getTrackbarPos('H Rho',         "Controls")
	HTheta    = (cv2.getTrackbarPos('H Theta',       "Controls") * np.pi / 360) + 0.001
	HThresh   = cv2.getTrackbarPos('H Thresh',      "Controls")
	HLen      = cv2.getTrackbarPos('H Len',         "Controls")
	HGap      = cv2.getTrackbarPos('H Gap',         "Controls")
	if (HThresh < 1):
		HThresh = 1
		cv2.setTrackbarPos('H Thresh',      "Controls", HThresh)
	if (HRho < 1):
		HRho = 1
		cv2.setTrackbarPos('H Rho',         "Controls", HRho)

def init():
	global CannyLow
	global CannyHigh
	global Blur
	global BlurSD
	global BiD
	global BiSigC
	global BiSigS
	global HRho
	global HTheta
	global HThresh
	global HLen
	global HGap

	#Create a window for display and to put the slider controls on it
	cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
	#Add sliders as necessary
	cv2.createTrackbar('Canny Low',     "Controls" , CannyLow,  255, EmptyCallback)
	cv2.createTrackbar('Canny High',    "Controls" , CannyHigh, 255, EmptyCallback)
	cv2.createTrackbar('Gauss Blur',    "Controls" , Blur,      20,  EmptyCallback)
	cv2.createTrackbar('Gauss SD',      "Controls" , BlurSD,    100, EmptyCallback)
	cv2.createTrackbar('H Rho',         "Controls" , HRho,      20,  EmptyCallback)
	cv2.createTrackbar('H Theta',       "Controls" , HTheta,    360, EmptyCallback)
	cv2.createTrackbar('H Thresh',      "Controls" , HThresh,   200, EmptyCallback)
	cv2.createTrackbar('H Len',         "Controls" , HLen,      100, EmptyCallback)
	cv2.createTrackbar('H Gap',         "Controls" , HGap,      100, EmptyCallback)

def alignment_detect(img):
	global CannyLow
	global CannyHigh
	global Blur
	global BlurSD
	global HRho
	global HTheta
	global HThresh
	global HLen
	global HGap

	#Convert the image to gray scale for simpler edge processing
	gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	gray_gaus_image = cv2.GaussianBlur(gray_image, (Blur, Blur), 0)
	thresh_gaus_image = cv2.threshold(gray_gaus_image, 127, 255, cv2.THRESH_BINARY)[1]


	edged_gaus_image = cv2.Canny(gray_gaus_image, CannyLow, CannyHigh)

#	contours = cv2.findContours(gray_bi_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#	contours = cv2.findContours(gray_bi_image_temp, 1, 2)

#	edged_bi_image_temp = np.copy(edged_bi_image)
#	contours,hierarchy = cv2.findContours(edged_bi_image_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#	print contours

#	cnt = contours[0]
#	cnt = max(contours, key = cv2.contourArea)
#	M = cv2.moments(cnt)

#	if M["m00"] != 0:
#	    cx = int(M["m10"] / M["m00"])
#	    cy = int(M["m01"] / M["m00"])
#	else:
#	    cx, cy = 0,0

#	print M
#	cx = int(M['m10']/M['m00'])
#	cy = int(M['m01']/M['m00'])
#	print "CX = ", cx, "CY = ", cy
#	cross_image = draw_cross(img, cx, cy)
#	epsilon = 0.1*cv2.arcLength(cnt,True)
#	approx = cv2.approxPolyDP(cnt,epsilon,True)

	#Recover rotated bounding box
#	rect = cv2.minAreaRect(cnt)
#	box = cv2.cv.BoxPoints(rect)
#	box = np.int0(box)
#	print box

#	contour_image = np.copy(img)

#	ConvexHullPoints = contoursConvexHull(contours)
#	print "HullPoints = ", ConvexHullPoints
#	cv2.polylines(contour_image, [ConvexHullPoints], True, (0,255,255), 2)

#	cv2.drawContours(contour_image,[box],0,(0,0,255),2) #this works
#	cv2.drawContours(contour_image, contours, 1, (0,255,0), 3)

#	epsilon = 0.1*cv2.arcLength(cnt,True)
#	approx = cv2.approxPolyDP(cnt,epsilon,True)

#	print "cnt = ", cnt
#	M = cv2.moments(cnt)
#	print M
#	print contours
#	cv2.drawContours(img, contours, -1, (0,255,0), 3)
#	cnt = contours[4]
#	cv2.drawContours(img, [cnt], 0, (0,255,0), 3)

#	#Dialate Cannyed image
#	kernel = np.ones((3,3), np.uint8)
#	dilated_image = cv2.dilate(cannyed_image, kernel, iterations=1)

#	#Crop the image to the triangle we defined
#	#Retrieve the image dimensions
#	height, width = img.shape[:2]
#	#Define a triangular region
#	region_of_interest_vertices = [ (0, height - 1), (width / 2, 0), (width - 1, height - 1), ]
#	cropped_image = region_of_interest( cannyed_image, np.array([region_of_interest_vertices], np.int32))

	#HaughLinesP version
#	plines_bi = cv2.HoughLinesP( edged_bi_image, rho=HRho, theta= HTheta, threshold=HThresh, lines=np.array([]), minLineLength=HLen, maxLineGap=HGap)
	plines_gaus = cv2.HoughLinesP( edged_gaus_image, rho=HRho, theta= HTheta, threshold=HThresh, lines=np.array([]), minLineLength=HLen, maxLineGap=HGap)
	#Add the lines to the original gray scale image
#	plines_bi_image = draw_lines(img, plines_bi)
	plines_gaus_image = draw_lines(img, plines_gaus)
	

	#HaughLines version
#	lines_bi = cv2.HoughLines( edged_bi_image,HRho, HTheta, HThresh)
	lines_gaus = cv2.HoughLines( edged_gaus_image,HRho, HTheta, HThresh)
	#Add the lines to the original gray scale image
#	lines_bi_image = draw_hough_lines(img, lines_bi)
	lines_gaus_image = draw_hough_lines(img, lines_gaus)

	#Display the lines found coordinate sets


	cv2.imshow("Source",            img)
	cv2.imshow("gray_image",        gray_image)
	cv2.imshow("gray_gaus_image",   gray_gaus_image)
	cv2.imshow("thresh_gaus_image", thresh_gaus_image)
#	cv2.imshow("edged_gaus_image",  edged_gaus_image)
#	cv2.imshow("lines_gaus_image",  lines_gaus_image)
#	cv2.imshow("plines_bi_image",   plines_bi_image)
#	cv2.imshow("plines_gaus_image", plines_gaus_image)
#	cv2.imshow("contour_image",     contour_image)
#	cv2.imshow("cross_image",       cross_image)

def contoursConvexHull(contours):
    pts = []
    for i in range(0, len(contours)):
        for j in range(0, len(contours[i])):
            pts.append(contours[i][j])

    pts = np.array(pts)
    result = cv2.convexHull(pts)
    return result

#This function will crop out all but a triangular region of the image
def region_of_interest(img, vertices):
	mask = np.zeros_like(img)
	match_mask_color = 255 # <-- This line altered for grayscale.
	cv2.fillPoly(mask, vertices, match_mask_color)
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def draw_only_lines(img, lines, color=[255, 0, 0], thickness=3):

	# Create a blank image that matches the original in size.
	line_img = np.zeros( ( img.shape[0], img.shape[1], 3), dtype=np.uint8,)

	# If there are no lines to draw, exit.
	if lines is None:
		return line_img

	# Loop over all lines and draw them on the blank image.
	for line in lines:
		for x1, y1, x2, y2 in line:
			cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

	# Return the lines image.
	return line_img

def draw_cross(img, x, y, color=[255, 0, 0], thickness=3):
	height, width, channels = img.shape 
	# Make a copy of the original image.
	img_out = np.copy(img)
	cv2.line(img_out, (0, y), (width, y), color, thickness)
	cv2.line(img_out, (x, 0), (x, height), color, thickness)
	return img_out

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
	# If there are no lines to draw, exit.
	if lines is None:
		return img

	# Make a copy of the original image.
	img = np.copy(img)

	# Create a blank image that matches the original in size.
	line_img = np.zeros( ( img.shape[0], img.shape[1], 3), dtype=np.uint8,)

	# Loop over all lines and draw them on the blank image.
	for line in lines:
		for x1, y1, x2, y2 in line:
			cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

	# Merge the image with the lines onto the original.
	img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

	# Return the modified image.
	return img

def draw_hough_lines(img, lines, color=[255, 0, 0], thickness=3):
#'lines' sre theta/rho as returned from HoughLines, NOT x1,y1,x2,y2 are returned from HoughLinesP
	# If there are no lines to draw, exit.
	if lines is None:
		return img
	#Otherwise draw the lines
	# Make a copy of the original image.
	img = np.copy(img)
	# Create a blank image that matches the original in size.
	line_img = np.zeros( ( img.shape[0], img.shape[1], 3), dtype=np.uint8,)
	for rho,theta in lines[0]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))

		cv2.line(line_img,(x1,y1),(x2,y2),(0,0,255),2)
	# Merge the image with the lines onto the original.
	out_img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
	# Return the modified image.
	return out_img


def do_processing():
	global img

	#Load an image from a file into 'img' array
#	img = cv2.imread("Alignment1.bmp");
#	img = cv2.imread("P1.jpg");
	img = cv2.imread("image001m.jpg");

	while True:
#		#Capture an image from the camera and assign to 'img' array
#		ret_val, img = cam.read()

		#Update parameters from the sliders
		UpdateParameters()

		#Pass the image in to the alignment detector
		alignment_detect(img)

		if cv2.waitKey(1) == 27: 
			break  # esc to quit


def main():
	init()
	do_processing()
	#Cleanup the display windows
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()


