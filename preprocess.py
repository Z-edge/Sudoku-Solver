import cv2 as cv


# Function to preprocess the input image
##############################################################################
def process(image):
    grayed_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred_img = cv.GaussianBlur(grayed_img, (7, 7), 0)
    thresh = cv.adaptiveThreshold(blurred_img, 255,
                                  cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY, 9, 2)
    thresh_inv = cv.bitwise_not(thresh)
    processed_image = thresh_inv
    # processed_image = cv.erode(thresh_inv,
    #                            cv.getStructuringElement(cv.MORPH_CROSS,
    #                                                     (3, 3)))
    # processed_image = cv.dilate(processed_img,
    #                             cv.getStructuringElement(cv.MORPH_CROSS,
    #                                                      (3, 3)))
    return processed_image
##############################################################################
