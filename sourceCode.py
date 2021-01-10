import cv2 as cv
from tensorflow .keras.models import load_model
import preprocess as pp
import puzzleFinder as pf
import boardMaker as bm
import sudokuSolver as ss
import getopt
import sys


# Managing command line arguments
##############################################################################
arg_list = sys.argv[1:]

options = 'm:n:sip'

long_options = ['model=', 'name=', 'showBoards',
                'intermediateDigits', 'showProcessingSteps']

model_name = ''
input_name = ''
debug = False
debug2 = False
debug3 = False

try:
    args, values = getopt.getopt(arg_list, options, long_options)

    for curr_arg, curr_value in args:
        if curr_arg in ('-m', '--model'):
            model_name = curr_value
        elif curr_arg in ('-n', '--name'):
            input_name = curr_value
        elif curr_arg in ('-s', '--showBoards'):
            debug = True
        elif curr_arg in ('-i', '--intermediateDigits'):
            debug2 = True
        elif curr_arg in ('-p', '--showProcessingSteps'):
            debug3 = True

except getopt.error as err:
    print(str(err))
##############################################################################

# Reading the image, processing it and transform it to extract the puzzle
##############################################################################
img = cv.imread('Inputs/' + input_name)

img = cv.resize(img, (600, 800),
                interpolation=cv.INTER_AREA)

processed_img = pp.process(img)
contour_img, required_contour = pf.locate_puzzle(processed_img.copy(),
                                                 img.copy())
final_colored, dst, src = pf.transform(required_contour, img)
final_grayed = cv.cvtColor(final_colored, cv.COLOR_BGR2GRAY)
##############################################################################

# Displaying the results
##############################################################################
cv.imshow("Original", img)

if debug3:
    cv.imshow('Processed', processed_img)
    cv.imshow('Required_Contour', contour_img)
    cv.imshow('Final', final_colored)
##############################################################################

# Loading the pre-trained model and making the sudoku board
##############################################################################
model = load_model('Models/' + model_name)
board, cell_locations = bm.makeBoard(model, final_grayed, debug2)
##############################################################################

# Printing unsolved and solved boards
##############################################################################
if debug:
    print(board)

solved_board, output = ss.solve(board, final_colored, img, dst, src, debug3)
if debug:
    print()
    print(solved_board)
##############################################################################

# Displaying the final Output
##############################################################################
cv.imshow('Output', output)
cv.waitKey(0)
##############################################################################
