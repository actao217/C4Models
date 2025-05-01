import matplotlib.pyplot as plt
from tqdm import trange
import random
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# === Board and Game Definitions ===
class Board:
    rows = 8
    cols = 9
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
        # Connect-5: need 5 in a row
        return (
            self._count(col, row, 1, 0, piece) + self._count(col, row, -1, 0, piece) - 1 >= 5 or
            self._count(col, row, 0, 1, piece) + self._count(col, row, 0, -1, piece) - 1 >= 5 or
            self._count(col, row, 1, 1, piece) + self._count(col, row, -1, -1, piece) - 1 >= 5 or
            self._count(col, row, 1, -1, piece) + self._count(col, row, -1, 1, piece) - 1 >= 5
        )

    def _count(self, col, row, dc, dr, piece):
        c, r = col, row
        count = 0
        while 0 <= c < self.cols and 0 <= r < self.rows and self.board[r][c] == piece:
            count += 1
            c += dc
            r += dr
        return count

    def get_valid_moves(self):
        return [c for c in range(self.cols) if self.board[self.rows-1][c] == Board.empty]

    def clone(self):
        new_b = Board()
        new_b.board = [row[:] for row in self.board]
        return new_b

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

# === Neural Network Definition ===
class NetworkB:
    def __init__(self, learning_rate=1e-3):
        self.model = models.Sequential([
            layers.Input(shape=(Board.rows, Board.cols, 1)),
            layers.Conv2D(128, (5, 5), activation='relu'),  # 5x5 filter
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['accuracy'])

    def predict(self, boards):
        boards = boards[..., np.newaxis]
        return self.model(boards, training=False).numpy()

    def train_on_batch(self, boards, targets):
        boards = boards[..., np.newaxis]
        return self.model.train_on_batch(boards, targets)

    def save(self, filepath="networkB_connect5.h5"):
        self.model.save(filepath)

    def load(self, filepath="networkB_connect5.h5"):
        self.model = tf.keras.models.load_model(filepath)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['accuracy'])

# === Reinforcement Learning Strategy ===
class NNStrategy:
    def __init__(self, network, player_id, discount_factor=0.9,
                 exploration_rate=0.1, batch_size=100):
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

# === Monte Carlo Tree Search Strategy ===
class Node:
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.visits = 0
        self.score = 0
        self.children = None

    def is_leaf(self): return self.children is None or self.children == []
    def is_terminal(self): return self.children == []

class Tree:
    def __init__(self): self.root = Node(None, None)

class MctsStrategy:
    def __init__(self, rollout_limit=150):
        self.rollout_limit = rollout_limit

    def move(self, game, player_id):
        tree = Tree()
        for _ in range(self.rollout_limit):
            self.__simulate_game(game.clone(), tree, player_id)
        best_child = max(tree.root.children, key=lambda c: c.visits)
        return best_child.action[1]

    def __simulate_game(self, game, tree, player_id):
        node = tree.root
        path = [node]
        # Selection
        while not node.is_leaf():
            node = self.__get_child_ucb(node)
            move_player, mv = node.action
            game.current_player = move_player
            game.move(mv)
            path.append(node)
        # Expansion
        if not game.finished:
            self.__add_children(node, game.board, game.current_player)
            node = self.__get_child_ucb(node)
            game.current_player = node.action[0]
            game.move(node.action[1])
            path.append(node)
            # Simulation
            while not game.finished:
                mv = random.choice(game.board.get_valid_moves())
                game.move(mv)
        # Backpropagation
        score = 1 if game.winner==player_id else 0 if game.winner is None else -1
        for nd in path:
            nd.visits += 1
            val = score if (nd.action is None or nd.action[0]==player_id) else -score
            nd.score += val

    def __add_children(self, parent, board, player_id):
        parent.children = [Node(parent, (player_id, mv)) for mv in board.get_valid_moves()]

    def __get_child_ucb(self, node):
        best = None
        best_ucb = -math.inf
        for c in node.children:
            u = (math.inf if c.visits==0 else c.score/c.visits +
                 2*math.sqrt(math.log(node.visits)/c.visits))
            if u>best_ucb:
                best_ucb, best = u, c
        return best

# === Helpers ===
def get_exploration_rate(game_number):
    return 0.1 * math.pow(1 - 0.000002, game_number)

def print_board(board_matrix):
    symbols = {1:'X', -1:'O', 0:'.'}
    for row in reversed(board_matrix):
        print('|' + ''.join(symbols[val] for val in row) + '|')
    print('+' + '-'*Board.cols + '+\n')

# === Training Loop ===
def train_networkB_vs_mcts150(num_games=100000, batch_size=100, block_size=500):
    netB = NetworkB(learning_rate=1e-3)
    agent = NNStrategy(netB, player_id='X', discount_factor=0.9,
                       exploration_rate=0.2, batch_size=batch_size)
    opponent = MctsStrategy(rollout_limit=150)

    # Metrics
    win_rates, draw_rates = [], []
    first_win_rates, second_win_rates = [], []
    x_vals = []

    # Block counters
    total_wins = draws = 0
    first_games = second_games = 0
    first_wins = second_wins = 0

    # Sample games
    sample_win, sample_loss = [], []

    for i in trange(num_games, desc="Training"):
        game = Game()
        game.current_player = random.choice(['X','O'])
        agent.exploration_rate = get_exploration_rate(i)
        moved_first = (game.current_player=='X')
        snapshots = []

        while not game.finished:
            snapshots.append(agent.encode_board_relative(game.board.clone()))
            if game.current_player=='X':
                agent.record_state(game.clone())
                mv = agent.pick_move(game)
            else:
                mv = opponent.move(game.clone(), 'O')
            game.move(mv)

        agent.game_over(game.winner,
                        last_player=('O' if game.current_player=='X' else 'X'))

        # Update stats
        if game.winner=='X':
            total_wins += 1
            if moved_first: first_wins += 1
            else:          second_wins += 1
        elif game.winner is None:
            draws += 1
        if moved_first: first_games += 1
        else:           second_games += 1

        # Block end
        if (i+1) % block_size == 0:
            win_rates.append(total_wins/block_size)
            draw_rates.append(draws/block_size)
            first_win_rates.append(first_wins/first_games if first_games else 0)
            second_win_rates.append(second_wins/second_games if second_games else 0)
            x_vals.append(i+1)

            print(f"\n--- Block {i+1} ---")
            print(f"Wins: {total_wins}/{block_size} ({win_rates[-1]:.2%}), "
                  f"Draws: {draws}/{block_size} ({draw_rates[-1]:.2%})")
            print(f" First→Wins: {first_wins}/{first_games} "
                  f"({first_win_rates[-1]:.2%}), "
                  f"Second→Wins: {second_wins}/{second_games} "
                  f"({second_win_rates[-1]:.2%})")

            # reset
            total_wins = draws = 0
            first_games = second_games = 0
            first_wins = second_wins = 0

    # Plot metrics
    plt.figure(figsize=(10,6))
    plt.plot(x_vals, win_rates,       label="Win Rate",             linewidth=2)
    plt.plot(x_vals, draw_rates,      label="Draw Rate",            linewidth=2)
    plt.plot(x_vals, first_win_rates, label="First‐Player Win Rate", linewidth=2)
    plt.plot(x_vals, second_win_rates,label="Second‐Player Win Rate",linewidth=2)
    plt.xlabel("Games Played")
    plt.ylabel(f"Rate (per {block_size}-game block)")
    plt.title("Connect-5 (8×9) NetworkB vs MCTS150 Bot")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("connect5_stats.png")
    plt.show()

    print("\n🔍 Sample game the model WON")
    if sample_win:
        print_board(sample_win[-1])
    else:
        print("No sample win recorded.")

    print("\n🔍 Sample game the model LOST")
    if sample_loss:
        print_board(sample_loss[-1])
    else:
        print("No sample loss recorded.")

    return netB

if __name__ == "__main__":
    model = train_networkB_vs_mcts150()
    model.save()
