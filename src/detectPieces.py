import cv2 as cv
import numpy as np

#Load image
image = cv.imread('pieces.jpg')
##resize image so that it fits the screen
resized = cv.resize(image, (0,0), fx=0.3, fy=0.3) ##leave out the dimensions for the desired output parameter, use scaling factors instead
#cv.imshow('Image', resized)
#cv.waitKey(0)

#grayscale conversion
gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
##convert to binary image using thresholding
thresh = cv.threshold(gray, 70, 255, cv.THRESH_BINARY_INV) #if pixel value greater than threshold, set to 0 , otherwise se to the maxval parameter (255 in this case)
cv.imshow('Gray Image', thresh[1])
cv.waitKey(0)

#contour detection -> use to detect pieces on the gameboard
#contours = list of contour points for each detected contour
#hierarchy = information about the contour hierarchy (parent-child relationships)/relationships between contours like nested or child contours
contours, hierarchy = cv.findContours(thresh[1], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #only get outer contours so that we can get top-level contours to detect pieces
#for piece detection, we only need outer contours, ignore the hierarchy
#Draw contours on the original image
contour_image = resized.copy()
cv.drawContours(contour_image, contours, -1, (255, 0, 0), 2) #draw all contours in blue, -1 means draw all contours, for line thickness use positive integer
#cv.imshow('Contours', contour_image)
#cv.waitKey(0)
