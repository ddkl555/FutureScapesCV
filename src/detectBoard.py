import cv2 as cv
import numpy as np

#Load image
image = cv.imread('gameboard.png')
#cv.imshow('Image', image)
#cv.waitKey(0)

#grayscale conversion
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray Image', gray)
#cv.waitKey(0)

#edge detection
edges = cv.Canny(gray, 50, 150)
cv.imshow('Edges', edges)
cv.waitKey(0)

#contour detection
contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# Draw contours on the original image
contour_image = image.copy()
cv.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
#cv.imshow('Contours', contour_image)
#cv.waitKey(0)

#find the largest contour
largest_contour = max(contours, key=cv.contourArea)
# Draw the largest contour on the original image
largest_contour_image = image.copy()
cv.drawContours(largest_contour_image, [largest_contour], -1, (0, 0, 255), 3)
#cv.imshow('Largest Contour', largest_contour_image)
#cv.waitKey(0)


