def valid_row(sudoku, val, row, col):
    invalid_row_indices = []
    for i in range(9):
        if i == col:
            continue
        if sudoku[row][i] == val:
            invalid_row_indices.append((row, i))
    return invalid_row_indices


def valid_col(sudoku, val, row, col):
    invalid_col_indices = []
    for i in range(9):
        if i == row:
            continue
        if sudoku[i][col] == val:
            invalid_col_indices.append((i, col))
    return invalid_col_indices


def valid_block(sudoku, val, row, col):
    row_bound = row - row % 3
    col_bound = col - col % 3
    invalid_block_indices = []
    for i in range(row_bound, row_bound+3):
        for j in range(col_bound, col_bound+3):
            if i == row and j == col:
                continue
            if sudoku[i][j] == val:
                invalid_block_indices.append((i, j))
    return invalid_block_indices


def validate_sudoku(sudoku):
    invalid_indices = set()
    for i in range(0, 9):
        for j in range(0, 9):
            val = sudoku[i][j]
            # Ignoring all empty values denoted by a zero
            if not val:
                continue
            for inv in valid_row(sudoku, val, i, j):
                invalid_indices.add(inv)
            for inv in valid_col(sudoku, val, i, j):
                invalid_indices.add(inv)
            for inv in valid_block(sudoku, val, i, j):
                invalid_indices.add(inv)
    return invalid_indices


def hints_sudoku(sudoku):
    hints_dict = {}
    for i in range(0, 9):
        string_val = 'ABCDEFGHI'
        for j in range(0, 9):
            if sudoku[i][j]:
                continue
            for val in range(1, 10):
                if not valid_row(sudoku, val, i, j) and not valid_col(sudoku, val, i, j) and not valid_block(sudoku, val, i, j):
                    if hints_dict.has_key((string_val[i], j+1)): 
                        hints_dict[string_val[i], j+1].append(val)
                    else:
                        hints_dict[string_val[i], j+1] = [val]
    return hints_dict

if __name__ == "__main__":
    sudoku = [
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
    # print(validate_sudoku(sudoku))
    mydict = hints_sudoku(sudoku)
    for key in sorted(mydict.iterkeys()):
        print "%s: %s" % (key, mydict[key])
