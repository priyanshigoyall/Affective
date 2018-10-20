"""Wrapper for sudoku validation and hints"""
class Sudoku(object):
    """Class for allowing user to print and validate a sudoku"""
    def __init__(self, sudoku):
        """Initialized sudoku class with a 9x9 matrix"""
        self.sudoku = sudoku


    def _valid_row(self, val, row, col, known_invalid):
        """Internal function to validate value for row"""
        for i in range(9):
            # If the pair(row,col) is already known to be invalid or comparing the value to self, ignore the pair
            if i == col or (row, i) in known_invalid:
                continue
            if self.sudoku[row][i] == val:
                known_invalid.add((row, i))
        return len(known_invalid)


    def _valid_col(self, val, row, col, known_invalid):
        """Internal function to validate value for column"""
        for i in range(9):
            # If the pair(row,col) is already known to be invalid or comparing the value to self, ignore the pair
            if i == row or (i, col) in known_invalid:
                continue
            if self.sudoku[i][col] == val:
                known_invalid.add((i, col))
        return len(known_invalid)


    def _valid_block(self, val, row, col, known_invalid):
        """Internal function to validate value for block"""
        row_bound = row - row % 3
        col_bound = col - col % 3
        for i in range(row_bound, row_bound+3):
            for j in range(col_bound, col_bound+3):
                # If the pair(row,col) is already known to be invalid or comparing the value to self, ignore the pair
                if i == row and j == col or (i, j) in known_invalid:
                    continue
                if self.sudoku[i][j] == val:
                    known_invalid.add((i, j))
        return len(known_invalid)


    def validate_sudoku(self):
        """Function to validate sudoku expressed as a list of lists"""
        if len(self.sudoku) != 9 or len(self.sudoku[0]) != 9:
            print "sudoku matrix must be of dimensions 9x9"
            return "all"
        invalid_indices = set()
        for i in range(0, 9):
            for j in range(0, 9):
                val = self.sudoku[i][j]
                # Ignoring all empty values denoted by a zero
                if not val:
                    continue
                # Ignoring all values not in range (1-9)
                if val not in range(1, 10):
                    invalid_indices.add((i, j))
                self._valid_row(val, i, j, invalid_indices)
                self._valid_col(val, i, j, invalid_indices)
                self._valid_block(val, i, j, invalid_indices)
        return sorted(invalid_indices)


    def _hints_sudoku(self):
        """Internal function to generate hints for sudoku"""
        hints_dict = {}
        string_val = 'ABCDEFGHI'
        for i in range(0, 9):
            for j in range(0, 9):
                if self.sudoku[i][j]:
                    continue
                for val in range(1, 10):
                    if not self._valid_row(val, i, j, set()) and \
                        not self._valid_col(val, i, j, set()) and \
                        not self._valid_block(val, i, j, set()):
                        if hints_dict.has_key((string_val[i], j+1)):
                            hints_dict[string_val[i], j+1].append(val)
                        else:
                            hints_dict[string_val[i], j+1] = [val]
        return hints_dict


    def print_hints(self):
        """Function to print hints of a valid self.sudoku"""
        validate_sudoku_val = self.validate_sudoku()
        if validate_sudoku_val:
            print "Invalid sudoku, please verify the following indices:" +\
                str(validate_sudoku_val)
            return
        hints_dict = self._hints_sudoku()
        for key in sorted(hints_dict.iterkeys()):
            print "%s: %s" % (key, hints_dict[key])

        return hints_dict


def example_matrix_test():
    # Verified that values from the example exactly match the output
    sudoku_test_example = [
        [7, 0, 0, 0, 8, 0, 0, 0, 2],
        [0, 0, 3, 0, 2, 4, 9, 0, 0],
        [0, 4, 0, 0, 0, 9, 0, 0, 0],
        [0, 8, 4, 2, 1, 0, 0, 0, 5],
        [0, 0, 9, 0, 0, 0, 2, 0, 0],
        [1, 0, 0, 0, 9, 5, 4, 3, 0],
        [0, 0, 0, 4, 0, 0, 0, 5, 0],
        [0, 0, 1, 6, 5, 0, 3, 0, 0],
        [2, 0, 0, 0, 3, 0, 0, 0, 4],
    ]
    sudoku_obj = Sudoku(sudoku_test_example)
    sudoku_obj.print_hints()


def negative_matrix_repeated_values():
    # Verified that values from the example exactly match the output
    sudoku_test_example = [
        [7, 0, 0, 0, 0, 0, 0, 0, 7],
        [0]*9,
        [0, 0, 7, 0, 0, 0, 0, 0, 0],
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
        [7, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    print "Entering repeated values in (0,0), (0,8), (2,2), (8,0)"
    sudoku_obj = Sudoku(sudoku_test_example)
    sudoku_obj.print_hints()


def negative_matrix_wrong_dimensions():
    sudoku_test_example = [
        [0]*20,
        [0]*20,
    ]
    print "Entering wrong dimensions of 20x2"
    sudoku_obj = Sudoku(sudoku_test_example)
    sudoku_obj.print_hints()


def negative_matrix_invalid_range_values():
    # Verified that values from the example exactly match the output
    sudoku_test_example = [
        [-1, 0, 10, 0, 0, 0, 0, 0, 100],
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
    ]
    print "Entering wrong range of values in (0,0), (0, 2), (0,8)"
    sudoku_obj = Sudoku(sudoku_test_example)
    sudoku_obj.print_hints()


def negative_matrix_inconsistent_type_values():
    # Verified that values from the example exactly match the output
    sudoku_test_example = [
        ['a', 0, 2.3, 0, 0, 0, 0, 0, [1,2,3,4]],
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
    ]
    print "Entering wrong range of values in (0,0), (0, 2), (0,8)"
    sudoku_obj = Sudoku(sudoku_test_example)
    sudoku_obj.print_hints()


def self_check_matrix_all_empty_test():
    # Verified that values from the example exactly match the output
    sudoku_test_example = [
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
        [0]*9,
    ]
    sudoku_obj = Sudoku(sudoku_test_example)
    return_dict = sudoku_obj.print_hints()
    for values in return_dict.itervalues():
        assert values == range(1, 10)


def test_wrapper():
    example_matrix_test()
    # Comment out the following function calls to check custom test cases, edit the value in example_matrix_test
    self_check_matrix_all_empty_test()
    negative_matrix_repeated_values()
    negative_matrix_wrong_dimensions()
    negative_matrix_invalid_range_values()
    negative_matrix_inconsistent_type_values()

if __name__ == "__main__":
    test_wrapper()
