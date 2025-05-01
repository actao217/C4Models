import numpy as np
import random

class NNStrategy:
    def __init__(self, network, player_id, discount_factor=0.9, exploration_rate=0.1, batch_size=100):
        self.network = network
        self.player_id = player_id
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.batch_size = batch_size
        self.current_game_states = []
        self.train_states = []
        self.train_targets = []

    def pick_move(self, game):
        valid_moves = game.board.get_valid_moves()
        if random.random() < self.exploration_rate:
            return random.choice(valid_moves)
        states_batch = []
        for mv in valid_moves:
            clone_g = game.clone()
            clone_g.move(mv)
            enc = self.encode_board_relative(clone_g.board)
            states_batch.append(enc)
        states_arr = np.array(states_batch, dtype=np.float32)
        values = self.network.predict(states_arr).flatten()
        best_idx = np.argmax(values)
        return valid_moves[best_idx]

    def record_state(self, game):
        enc = self.encode_board_relative(game.board)
        self.current_game_states.append(enc)

    def encode_board_relative(self, board):
        opp = 'O' if self.player_id == 'X' else 'X'
        return np.array([
            [1 if cell == self.player_id else -1 if cell == opp else 0 for cell in row]
            for row in board.board
        ], dtype=np.float32)

    def game_over(self, winner, last_player):
        if winner is None:
            reward = 0
        elif winner == self.player_id:
            reward = 1
        else:
            reward = -1
        for state in reversed(self.current_game_states):
            self.train_states.append(state)
            self.train_targets.append(reward)
            reward *= self.discount_factor
        self.current_game_states.clear()
        if len(self.train_targets) >= self.batch_size:
            states_arr = np.array(self.train_states, dtype=np.float32)
            targets_arr = np.array(self.train_targets, dtype=np.float32)
            self.network.train_on_batch(states_arr, targets_arr)
            self.train_states.clear()
            self.train_targets.clear()

    def save(self, filepath="networkB.h5"):
        self.network.save(filepath)