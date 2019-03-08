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
	cv2.createTrackbar('Bi D',          "Controls" , BiD,       20,  EmptyCallback)
	cv2.createTrackbar('Bi SigC',       "Controls" , BiSigC,    255, EmptyCallback)
	cv2.createTrackbar('Bi SigS',       "Controls" , BiSigS,    10,  EmptyCallback)
	cv2.createTrackbar('H Rho',         "Controls" , HRho,      20,  EmptyCallback)
	cv2.createTrackbar('H Theta',       "Controls" , HTheta,    360, EmptyCallback)
	cv2.createTrackbar('H Thresh',      "Controls" , HThresh,   200, EmptyCallback)
	cv2.createTrackbar('H Len',         "Controls" , HLen,      100, EmptyCallback)
	cv2.createTrackbar('H Gap',         "Controls" , HGap,      100, EmptyCallback)

def alignment_detect(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	gray = np.float32(gray)
	dst1 = cv2.cornerHarris(gray,2,3,0.04)

	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst1,None)

	# Threshold for an optimal value, it may vary depending on the image.
	img[dst>0.01*dst1.max()]=[0,0,255]

	cv2.imshow('img',img)
#	cv2.imshow('dst',dst)
#	cv2.imshow('dst1',dst1)
	
def FindDistance(Point1, Point2):
	#Note, don't bother finding the square root of the distance since we only care about the relative sizes not the actual distances
	xd = abs(Point1[0] - Point2[0])
	yd = abs(Point1[1] - Point2[1])
	return (xd*xd) + (yd*yd)
	
def FindCenter(Point1, Point2):
	NewX = (Point1[0] + Point2[0]) / 2
	NewY = (Point1[1] + Point2[1]) / 2
	return NewX, NewY

def ReducePoints(Points, TargetCount):
	
	#Convert to regular list so we can work with it more easily
	WorkingList = list(np.copy(Points))
	
	while True:
		MinimumDistance = 10000000
		#Get the current point count
		PointsCount = len(WorkingList)
		#If already correct or too low then exit
		if (PointsCount <= TargetCount):
			return WorkingList
			break
		#Otherwise cycle through all contour points and find the shortest distance pair
		#Note, don't bother finding the square root of the distance since we only care about the relative sizes not the actual distances
		for i in range(PointsCount):
			Distance = FindDistance(WorkingList[i][0], WorkingList[(i + 1) % PointsCount][0])
			if (Distance < MinimumDistance):
				MinimumDistance = Distance
				MinimumIndex = i
		#Found the minimum distance pair so find average position and replace 2 points with 1
		WorkingList[MinimumIndex][0] = FindCenter(WorkingList[MinimumIndex][0], WorkingList[(MinimumIndex + 1) % PointsCount][0])
		IndexToRemove = (MinimumIndex + 1) % PointsCount
		WorkingList.pop(IndexToRemove)
	return WorkingList

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

def skeleton(img):
	size = np.size(img)
	skel = np.zeros(img.shape,np.uint8)
#	ret,img = cv2.threshold(img,127,255,0)
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	done = False
	while( not done):
		eroded = cv2.erode(img,element)
		temp = cv2.dilate(eroded,element)
		temp = cv2.subtract(img,temp)
		skel = cv2.bitwise_or(skel,temp)
		img = eroded.copy()
		zeros = size - cv2.countNonZero(img)
		if zeros==size:
		        done = True
	return skel

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
#	img = cv2.imread("p1.jpg");
	img = cv2.imread("p1-white.jpg");

	while True:
#		#Capture an image from the camera and assign to 'img' array
#		ret_val, img = cam.read()

		#Update parameters from the sliders
		UpdateParameters()

		#Pass the image in to the alignment detector
		alignment_detect(img)

#		if cv2.waitKey(1) == 27: #(1) for video
		if cv2.waitKey(0) == 27: #(0) for static pic
			break  # esc to quit


def main():
	init()
	do_processing()
	#Cleanup the display windows
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()



