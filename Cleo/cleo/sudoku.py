def solve_sudoku(board):
    empty = _find_empty(board)
    if empty is None:
        return True

    row, col = empty
    for number in range(1, 10):
        if _is_valid(board, row, col, number):
            board[row][col] = number
            if solve_sudoku(board):
                return True
            board[row][col] = 0
    return False


def format_board(board):
    return "\n".join(" ".join(str(value) if value else "." for value in row) for row in board)


def demo_solution():
    board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    solve_sudoku(board)
    return format_board(board)


def _find_empty(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return row, col
    return None


def _is_valid(board, row, col, number):
    if any(board[row][c] == number for c in range(9)):
        return False
    if any(board[r][col] == number for r in range(9)):
        return False

    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == number:
                return False
    return True
