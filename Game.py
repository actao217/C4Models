from Board import Board

class Game:
    def __init__(self):
        self.board = Board()
        self.current_player = 'X'
        self.winner = None
        self.finished = False

    def move(self, col):
        if self.finished:
            raise ValueError('Game is already finished')
        if self.board.drop(col, self.current_player):
            self.winner = self.current_player
            self.finished = True
        else:
            if len(self.board.get_valid_moves()) == 0:
                self.finished = True
                self.winner = None
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def clone(self):
        g = Game()
        g.board = self.board.clone()
        g.current_player = self.current_player
        g.winner = self.winner
        g.finished = self.finished
        return g
