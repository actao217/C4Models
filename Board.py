class Board:
    rows = 6
    cols = 7
    empty = '.'

    def __init__(self):
        self.board = [[Board.empty for _ in range(self.cols)] for _ in range(self.rows)]

    def drop(self, col, piece):
        if not (0 <= col < self.cols):
            raise ValueError('Column out of bounds')
        for row in range(self.rows):
            if self.board[row][col] == Board.empty:
                self.board[row][col] = piece
                return self.check_win(row, col, piece)
        raise ValueError('Column is full')

    def check_win(self, row, col, piece):
        return (self._count(col, row, 1, 0, piece) + self._count(col, row, -1, 0, piece) - 1 >= 4 or
                self._count(col, row, 0, 1, piece) + self._count(col, row, 0, -1, piece) - 1 >= 4 or
                self._count(col, row, 1, 1, piece) + self._count(col, row, -1, -1, piece) - 1 >= 4 or
                self._count(col, row, 1, -1, piece) + self._count(col, row, -1, 1, piece) - 1 >= 4)

    def _count(self, col, row, dc, dr, piece):
        c, r = col, row
        count = 0
        while 0 <= c < self.cols and 0 <= r < self.rows and self.board[r][c] == piece:
            count += 1
            c += dc
            r += dr
        return count

    def get_valid_moves(self):
        return [c for c in range(self.cols) if self.board[self.rows - 1][c] == Board.empty]

    def clone(self):
        new_b = Board()
        new_b.board = [row[:] for row in self.board]
        return new_b