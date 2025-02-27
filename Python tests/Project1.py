#Developed from code at https://medium.com/@mrhwick/simple-lane-detection-with-opencv-bfeb6ae54ec0
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

def show_webcam():
	#Open the first video device and bind to 'cam'
	cam = cv2.VideoCapture(0)
	while True:
		#Capture an image from the camera and assign to 'img' array
		ret_val, img = cam.read()
		if (ret_val == 0):
			print'No camera device found/image not found'
			exit()
		#Retrieve the image dimensions
		height, width = img.shape[:2]
		#Define a triangular region
		region_of_interest_vertices = [ (0, height - 1), (width / 2, 0), (width - 1, height - 1), ]
		#Convert the image to gray scale for simpler edge processing
		gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		#Do edge detection on the image now
		cannyed_image = cv2.Canny(gray_image, 100, 200)
		#Crop the image to the triangle we defined
		cropped_image = region_of_interest( cannyed_image, np.array([region_of_interest_vertices], np.int32))

		lines = cv2.HoughLinesP( cropped_image, rho=6, theta=np.pi / 60, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=25)
		#Display the lines found coordinate sets
		#print(lines)
		#Add the lines to the original image
		ne_image = draw_lines(img, lines)

		cv2.imshow('Vision', ne_image)

#		plt.figure()
		#'show' the cropped image to a pyplot surface 'plt'
#		plt.imshow(gray_image)
		#Render the pyplot surface 'plt' to a display window
#		plt.show()
		#Check if escape pressed and exit if yes (never reaches here from plt.show() !!)
		if cv2.waitKey(1) == 27: 
			break  # esc to quit
	#Cleanup the display windows
	cv2.destroyAllWindows()

#This function will crop out all but a triangular region of the image
def region_of_interest(img, vertices):
	mask = np.zeros_like(img)
	match_mask_color = 255 # <-- This line altered for grayscale.
	cv2.fillPoly(mask, vertices, match_mask_color)
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
	# If there are no lines to draw, exit.
	if lines is None:
		return

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

def main():
	show_webcam()

if __name__ == '__main__':
	main()
