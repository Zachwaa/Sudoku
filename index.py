import math
import time
import sys

from sudokus import MediumNYT130324, HardNYT130324, EasySudoku

# TODO Check if proper error handling is used if an invalid sudoku is passed in 
class Sudoku:
    def __init__(self, sudoku):
        # A row should look like this 
        # [1,2,3,4,5,6,7,8,9]

        if (len(sudoku) != 9):
            print (
                "Invalid Sudoku passed in.", 
                len(sudoku), 
                "rows passed in. Should be 9"
            )

        for index, row in enumerate(sudoku):
            if (len(row) != 9):
                print (
                    "Invalid Sudoku passed in.", 
                    len(row), 
                    "columns passed in on row" ,
                    index + 1, 
                    ". Should be 9"
                )

        self._checks = []
        self.sudoku = sudoku 
        self.count = 0

    def _parseSudoku(self, sudoku):
        columns = [[] for _ in range(9)]    
        grids = [[[], [], []] for _ in range(3)]
        rows = []

        for indexRow, row in enumerate(sudoku):
            # Get the values in each row
            rows.append(row)
            for indexCol, entity in enumerate(row):
                # Gets the values in each column
                columns[indexCol].append(entity)
                # Gets the values in each grid
                grids[indexRow // 3][indexCol // 3].append(entity)

        return rows, columns, grids
    
    def _findEmptySquares(self, sudoku):
        emptySquares = []
        for indexRow, row in enumerate(sudoku):
            for indexCol, col in enumerate(row):
                if not col:
                    emptySquares.append((indexRow, indexCol))

        return emptySquares
    
    def _getPossibilitiesForEverySquare(self, sudoku):
        valuesPerSquare = {}

        rows, columns, grids = self._parseSudoku(sudoku)
        emptySquares = self._findEmptySquares(sudoku)

        # Get all the valid values per empty square 
        for (row, col) in emptySquares:
            validValues = []

            for i in range(1,10):
                if (
                    i not in columns[col] and
                    i not in rows[row] and
                    i not in grids[row // 3][col // 3]
                ):
                    validValues.append(i)

            valuesPerSquare[(row, col)] = validValues

        return valuesPerSquare

    def _findSquareWithLeastPossibilities(self, sudoku):
        valuesPerSquare = self._getPossibilitiesForEverySquare(sudoku)

        lowest = 10000
        bestSquare = ()

        # Find the square with the least number of values
        # To speed up the process, if a square has one value then do that square

        for square, validValues in valuesPerSquare.items():
            if len(validValues) == 1:
                return (square, validValues)

            if len(validValues) < lowest:
                lowest = len(validValues)
                bestSquare = (square, validValues)

        return bestSquare
        
    def solve(self, sudoku = None):
        if not sudoku: 
            # TODO This is hack for now for the first iteration
            sudoku = self.sudoku

        self.count += 1

        time.sleep(0.02)

        for i in sudoku:
            print(i)

        print ("\n")

        if self._validateSudoku(sudoku):
            return sudoku

        square, validValues = self._findSquareWithLeastPossibilities(sudoku)
        row, col = square

        for number in validValues:
            sudoku[row][col] = number
            nestedSudoku = self.solve(sudoku)
            if nestedSudoku:
                # Found a valid solution so break recursion
                return nestedSudoku
            else:
                # Solution is not valid so undo move
                sudoku[row][col] = None

        # This means the path lead to the wrong solution
        # So backtrack to initial move and choose the other option
        return None

    def _validateOneToNine(self, array):
        for n in array:
            if not n:
                return False

        for i, number in enumerate(sorted(array)):
            if number != i + 1:
                return False
    
        return True
    
    def _validateSudoku(self, sudoku):
        if not sudoku:
            return False

        rows, columns, grids = self._parseSudoku(sudoku)

        checks = []

        grids_flat = [item for grid in grids for item in grid]
        checks.extend(grids_flat)
        checks.extend(rows)
        checks.extend(columns)

        for arr in checks:
            if not self._validateOneToNine(arr):
                return False

        return True

    @staticmethod          
    def getFormattedSudoku(sudoku):
        completedSudoku = ''
          
        for rowIndex, row in enumerate(sudoku):

            completedSudoku += "\n"
            if rowIndex != 0 and rowIndex % 3 == 0:
                completedSudoku += "-" * ((len(sudoku)) * 4)
                completedSudoku += "\n"

            for colIndex, number in enumerate(row):
                if colIndex != 0 and colIndex % 3 == 0:
                    completedSudoku += " | "

                if colIndex == 0:
                    completedSudoku += " "

                completedSudoku += f" {str(number)} "

        return completedSudoku
                
# The sudoku solver below uses a linear method
# It could only solve easy problems and so was replaced

class OldSudoku:
    def __init__(self, sudoku):
        # A row should look like this 
        # [1,2,3,4,5,6,7,8,9]

        if (len(sudoku) != 9):
            print (
                "Invalid Sudoku passed in.", 
                len(sudoku), 
                "rows passed in. Should be 9"
            )

        for index, row in enumerate(sudoku):
            if (len(row) != 9):
                print (
                    "Invalid Sudoku passed in.", 
                    len(row), 
                    "columns passed in on row" ,
                    index + 1, 
                    ". Should be 9"
                )

        self._checks = []
        self._state = [row[:] for row in sudoku]
        self.sudoku = sudoku 

    def _findValidValue(self, rowIndex, columnIndex, value):
        column = []
        grid = []
        row = self._state[rowIndex]

        for rowRange in range(0,9):
            # Get the values in the column
            column.append(self._state[rowRange][columnIndex])
                    
        upToIndexRow = (int(math.floor((rowIndex) / 3)) + 1) * 3
        upToIndexCol = (int(math.floor((columnIndex) / 3)) + 1) * 3
        
        # Get the values in the same grid
        for rowRange in range(upToIndexRow - 3, upToIndexRow):
            for col in range(upToIndexCol - 3, upToIndexCol):
                grid.append(self._state[rowRange][col])

        # Now try each possible value
        while value <= 10:
            if value == 10:
                # We have looped through all the possible values
                # So leave cell blank and go back 
                return " " 
            
            if (
                value not in column and 
                value not in row and 
                value not in grid
            ):
                return value     

            value += 1   

    def _next(self, row, col):
        if col + 1 < 9:
            return row, col + 1
        else:
            return row + 1, 0 

    def solveLinear(self, cell= (0,0), reverse = False):
        row, col = cell

        time.sleep(0.02)
        for i in self._state:
            print (i)

        print ("\n")

        if row == 9:
            return self._state
  
        activeCell = self._state[row][col]

        # If we are going back then we know the value is not fixed
        if not reverse and activeCell != ' ' and activeCell == self.sudoku[row][col]:
            # If value is fixed 
            return self.solveLinear(self._next(row, col))  

        value = 1
        if reverse:
            # Incrament the previous value and try again
            value = activeCell + 1

        # Try to find a valid value 
        cellValue = self._findValidValue(row,col,value)
        self._state[row][col] = cellValue

        if cellValue == ' ':
            # No valid value was found so go back
            try:
                lastCheck = self._checks[len(self._checks) - 1]
            except:
                return
      
            del self._checks[-1]
            return self.solveLinear(lastCheck, True) 
        else:
            # Valid value was found so move on to the next
            self._checks.append((row, col)) 
            return self.solveLinear(self._next(row, col)) 


NYTSudoku = Sudoku(MediumNYT130324)

print (Sudoku.getFormattedSudoku(NYTSudoku.solve()))