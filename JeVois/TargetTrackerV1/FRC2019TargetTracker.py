import libjevois as jevois
import cv2
import numpy as np
import math

class FRC2019TargetTracker:
    
    def __init__(self):
     
        self.timer = jevois.Timer("sandbox", 100, jevois.LOG_INFO)
        
    # ###################################################################################################

    ## Process function with USB output
    def process(self, inframe, outframe):
        
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR by default. If
        # you need a grayscale image instead, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB()
        # and getCvRGBA():
        inimg = inframe.getCvBGR()
        
        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()
        
        #Find the image size
        height, width, channels = inimg.shape
        
        target, outimg = self.alignment_detect(inimg)
                
        # Write a title:
        cv2.putText(outimg, "Robocats Vision Guide", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        cv2.putText(outimg, str(target[0]), (3, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        
        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        #Check output image size
        outheight, outwidth, outchannels = outimg.shape
        #Display some facts at the bottom of the image
        
        cv2.putText(outimg, fps, (3, outheight - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

        # Convert our OpenCv output image to video output format and send to host over USB:
        outframe.sendCv(outimg)

    def alignment_detect(self, sourceImg):
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

        #Run through contours and draw circles, adding coordinates and size to a list
        location, displayImg = self.findTargets(contours, sourceImg)
        #Convert from -1 +1 to window sized/centerd coordinates
        locationWindow = (int(((location[0] * width) + width) / 2) , int(((location[1] * height) + height) / 2))
        #Draw center line guide
        displayImg = self.draw_cross(displayImg, midX, midY, color=[255, 0, 0], thickness=1)
        #Draw target centers guide
        displayImg = self.draw_cross(displayImg, locationWindow[0], locationWindow[1], color=[140, 0, 200], thickness=2)
        return location, displayImg
        
    def findTargets(self, contours, imgIn):
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
        targetCenter = self.findCenterMid(blobLocations)
        return targetCenter, imgOut

    def findCenterMid(self, points):
        if (len(points) < 2):#Less than 2 points so can't really track markers. Keep going straight
            return (0, 0)
        elif (len(points) == 2): #Only 2 points found so just use them directly
            return self.findCenter(points[0], points[1])
        else:
            #Multiple points so need to find the closest 2 points to the center line
            entries = len(points)
            p1 = (10000, 10000)
            p2 = (10000, 10000)
            for loop in range(0, entries):
                p1, p2 = self.findSmallestTwo(points[loop], p1, p2)
        
        return self.findCenter(p1, p2)

    def findSmallestTwo(self, p1, p2, p3):
        #Check the x deviations from center and pick the smallest 2 from 3
        if (abs(p1[0]) < abs(p2[0])):         #p1 < p2, check where p3 is
            if (abs(p2[0]) < abs(p3[0])):
                return p1, p2
            else :
                return p1, p3
        else:                                 #p2 < p1, check where p3 is
            if (abs(p3[0]) < abs(p1[0])):
                return p2, p3
            else :
                return p2, p1

    def findCenter(self, p1, p2):
        return ((p1[0] + p2[0]) / 2),((p1[1] + p2[1]) / 2) 

    def draw_cross(self, img, x, y, color=[255, 0, 0], thickness=3):
        height, width, channels = img.shape 
        # Make a copy of the original image.
        img_out = np.copy(img)
        cv2.line(img_out, (0, y), (width, y), color, thickness)
        cv2.line(img_out, (x, 0), (x, height), color, thickness)
        return img_out

