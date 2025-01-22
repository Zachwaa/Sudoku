import cv2
import numpy as np
import pytesseract
import imutils
# import pyplot as plt
from PIL import Image

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped
        

# Function to preprocess the image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    thresh = cv2.bitwise_not(thresh)

    return thresh

def find_sudoku_grid(image):
    # Finds the Sudoku grid in the preprocessed image and extracts its contour.

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    thresh = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    puzzleCnt = None
      
    # loop over the contours
    for c in cnts:
        peri = cv2.arcLength(c, True) 
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        if len(approx) == 4:  
            puzzleCnt = approx
            break
        
    if puzzleCnt is None:
       raise Exception(("Could not find Sudoku puzzle outline. "
            "Try debugging your thresholding and contour steps."))

    output = image.copy()
    cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Puzzle Outline", output)
    cv2.waitKey(0)

    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

    # show the output warped image (again, for debugging purposes)
    cv2.imshow("Puzzle Transform", puzzle)
    cv2.waitKey(0)
    
    return (puzzle, warped)

# def extract_cells(image, grid):
#     # Warps the input image to extract individual cells from the Sudoku grid.
#     grid = np.array(grid, dtype="float32")
#     (tl, bl, br, tr) = grid

#     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

#     maxWidth = max(int(widthA), int(widthB))
#     maxHeight = max(int(heightA), int(heightB))

#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth - 1, maxHeight - 1],
#         [0, maxHeight - 1]], dtype="float32")

#     # Image.fromarray(dst).show()
#     test = cv2.resize(image,(maxWidth,maxHeight))

#     M = cv2.getPerspectiveTransform(grid, dst)  # Computes perspective transform matrix
#     warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))  # Applies perspective transformation

#     cv2.imshow("test", test)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     cell_height = maxHeight // 9
#     cell_width = maxWidth // 9
    
#     cells = []
#     for y in range(9):
#         row = []
#         for x in range(9):
#             cell = warped[y * cell_height:(y + 1) * cell_height, x * cell_width:(x + 1) * cell_width]
#             row.append(cell)
      
#         cells.append(row)

#     return cells

def remove_border(image, border_width):
    # Get image dimensions
    height, width = image.shape[:2]

    # Crop the image to remove the border
    cropped_image = image[border_width:height - border_width, border_width:width - border_width]

    return cropped_image

def extract_digit(cell, debug=False):
    # apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    # check to see if we are visualizing the cell thresholding step
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
        return None
    # otherwise, find the largest contour in the cell and create a
    # mask for the contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    # if less than 3% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
    if percentFilled < 0.03:
        return None
    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    # check to see if we should visualize the masking step
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)
    # return the digit to the calling function
    return digit

# def ocr_cells(cells):
#     # Performs OCR on the extracted cells to recognize digits.
#     sudoku_grid = []
#     for y, row in enumerate(cells):
#         row_digits = []
#         for x, cell in enumerate(row):
#             # Use pytesseract for OCR
#             # cv2.imshow(f"Cell ({x}, {y})", cell)
#             # cv2.waitKey(0)
#             # cv2.destroyAllWindows()
#             digit = pytesseract.image_to_string(cell, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
#             if digit.isdigit():
#                 row_digits.append(int(digit))
#             else:
#                 row_digits.append(0)  # If OCR fails, mark it as 0
#         sudoku_grid.append(row_digits)
#     return sudoku_grid

image = cv2.imread("./images/EasySudokuImage.png")

image = imutils.resize(image, width=600)
# preProcessedImage = preprocess_image(image)
(puzzle, warped) = find_sudoku_grid(image)

stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9

cellLocs = []

board = []

for y in range(0, 9):
    # initialize the current list of cell locations
    row = []
    for x in range(0, 9):
        # compute the starting and ending (x, y)-coordinates of the
        # current cell
        startX = x * stepX
        startY = y * stepY
        endX = (x + 1) * stepX
        endY = (y + 1) * stepY
        # add the (x, y)-coordinates to our cell locations list
        row.append((startX, startY, endX, endY))
          
        cell = warped[startY:endY, startX:endX]
        digit = extract_digit(cell, True)
        # verify that the digit is not empty
        if digit is not None:
            # resize the cell to 28x28 pixels and then prepare the
            # cell for classification
            roi = cv2.resize(digit, (28, 28))
            roi = roi.astype("float") / 255.0
            # roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            # classify the digit and update the Sudoku board with the
            # prediction
            pred = pytesseract.image_to_string(cell, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            board[y, x] = pred
    # add the row to our cell locations
    cellLocs.append(row)

print (board)


