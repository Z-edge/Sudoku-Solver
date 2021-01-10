import cv2 as cv
import numpy as np


# Function to locate the contour corresponding to
# the puzzle in the processed image
##############################################################################
def locate_puzzle(processed_image, image):
    contours = cv.findContours(processed_image,
                               cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    req_contour = None
    for c in contours:
        p = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.08 * p, True)
        if len(approx) == 4:
            req_contour = approx
            break
    contour_image = image
    cv.drawContours(contour_image, [req_contour], -1, (255, 0, 0), 3)
    return contour_image, req_contour
##############################################################################


# Function to transform the input image to obtain the warped puzzle
##############################################################################
def transform(contour, img):
    rect = contour.reshape([4, 2])
    summation = rect.sum(axis=1)
    difference = np.diff(rect, axis=1)

    t_l = rect[summation.argmin()]
    b_r = rect[summation.argmax()]
    t_r = rect[difference.argmax()]
    b_l = rect[difference.argmin()]
    src_rect = np.array([t_l, t_r, b_l, b_r], 'float32')

    def dist(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    sides = np.array([dist(t_l, t_r), dist(t_l, b_l),
                      dist(b_r, b_l), dist(b_r, t_r)])
    max_side = sides.max()
    dst_rect = np.array([[0, 0],
                         [0, max_side - 1],
                         [max_side - 1, 0],
                         [max_side-1, max_side-1]],
                        'float32')
    transformation_matrix = cv.getPerspectiveTransform(src_rect, dst_rect)
    f = cv.warpPerspective(img,
                           transformation_matrix,
                           (int(max_side), int(max_side)))
    return f, src_rect, dst_rect
##############################################################################
