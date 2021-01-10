import numpy as np
import cv2


# Function to check cells having zero value in the sudoku board
#############################################################################
def findEmpty(sudoku, N, M):
    for i in range(N):
        for j in range(M):
            if sudoku[i][j] == 0:
                return(i, j)
    return(-1, -1)
#############################################################################


# Function to check if a number can be inserted into the current cell
#############################################################################
def checking(sudoku, m, n, value, N, M):
    for i in range(M):
        if(sudoku[m][i] == value and i != n):
            return(False)
    for j in range(N):
        if(sudoku[j][n] == value and j != m):
            return(False)
    subx = m // 3
    suby = n // 3
    for i1 in range(subx*3, subx*3+3):
        for j1 in range(suby*3, suby*3+3):
            if(sudoku[i1][j1] == value and i1 != m and j1 != n):
                return(False)
    return(True)
#############################################################################


# Function to solve the puzzle using backtracking
#############################################################################
def sudokuSolver(sudoku, N, M):
    xy = findEmpty(sudoku, N, M)
    if(xy != (-1, -1)):
        for i in range(9):
            if(checking(sudoku, xy[0], xy[1], i+1, N, M)):
                sudoku[xy[0]][xy[1]] = i + 1
                if(sudokuSolver(sudoku, N, M)):
                    return(True)
                else:
                    sudoku[xy[0]][xy[1]] = 0
    else:
        return(True)
    return(False)
#############################################################################


# Function to display the solution augmented on the input image
#############################################################################
def displayAugmented(img, img2, sCopy, sudoku, dest, src, debug):
    N, M = sudoku.shape
    w = img.shape[0]//N
    h = img.shape[1]//M
    for x in range(N):
        for y in range(M):
            if(sCopy[y][x] == 0):
                cv2.putText(img, str(sudoku[y][x]),
                            (x*w+int(w/2)-8, int((y+0.9)*h)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                            (90, 50, 0),
                            2,
                            cv2.LINE_AA)
    if debug:
        cv2.imshow('Warped_Output', img)
    transformation_matrix = cv2.getPerspectiveTransform(src, dest)
    img = cv2.warpPerspective(img, transformation_matrix, (600, 800))
    for i1 in range(img.shape[0]):
        for j1 in range(img.shape[1]):
            if (np.array_equal([0, 0, 0], img[i1, j1])) is False:
                img2[i1, j1] = img[i1, j1]
    return(img2)
#############################################################################


# Runner function
#############################################################################
def solve(board, warped_img, original_img, dst, src, debug):
    sCopy = board.copy()
    N, M = board.shape
    sudokuSolver(board, N, M)
    return(board, displayAugmented(warped_img,
                                   original_img,
                                   sCopy,
                                   board, dst, src, debug))
#############################################################################
