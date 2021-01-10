import cv2 as cv
import numpy as np
from skimage.segmentation import clear_border


# Function to find digits in each cell of the puzzle image
##############################################################################
def find_digit(cell, x, y, debug):
    def is_noise(filtered_thresh):
        p = cv.countNonZero(filtered_thresh) / float(filtered_thresh.shape[0]
                                                     *
                                                     filtered_thresh.shape[1])
        if p < 0.03:
            return True
        return False

    thresh = cv.threshold(cell, 0, 255,
                          cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    if debug:
        cv.imshow('Threshed_digit' + str((x, y)), thresh)
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)[0]
    if len(contours) == 0:
        return None
    max_contour = max(contours, key=cv.contourArea)
    filtered_thresh = np.zeros(thresh.shape, dtype=np.uint8)
    cv.drawContours(filtered_thresh, [max_contour], -1, (255, 255, 255), -1)
    if is_noise(filtered_thresh):
        return None

    extracted_digit = cv.bitwise_and(thresh, thresh, mask=filtered_thresh)
    if debug:
        cv.imshow('Extracted_digit' + str((x, y)), extracted_digit)
    return extracted_digit
##############################################################################
