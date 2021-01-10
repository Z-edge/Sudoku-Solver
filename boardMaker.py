import cv2 as cv
import numpy as np
import digitFinder as df


# Function to make the numpy sudoku baord by
# predicting the extracted digits using the saved model
##############################################################################
def makeBoard(model, final_grayed, debug):
    board = np.zeros((9, 9), dtype=int)

    dx = final_grayed.shape[0] // 9
    dy = final_grayed.shape[1] // 9

    cell_locations = []

    for x in range(9):
        row = []
        for y in range(9):
            t_l = (x * dx, y * dy)
            b_r = ((x+1) * dx, (y+1) * dy)
            row.append((t_l, b_r))

            cell = final_grayed[t_l[0]: b_r[0], t_l[1]: b_r[1]]
            if debug:
                cv.imshow('Cropped_digit' + str((x, y)), cell)
            digit = df.find_digit(cell, x, y, debug)
            if digit is not None:
                r = cv.resize(digit, (28, 28))
                if debug:
                    cv.imshow('Resized_digit' + str((x, y)), r)
                r = r.reshape([1] + list(r.shape) + [1])
                r = r.astype('float32') / 255

                predicted_digit = model.predict(r).argmax(axis=1)[0]
                board[x, y] = predicted_digit
        cell_locations.append(row)
    return board, cell_locations
##############################################################################
