import matplotlib.pyplot as plt
from tqdm import trange
import random
import math

from NetworkB import NetworkB
from Game import Game
from NNStrategy import NNStrategy
from MCTS import MctsStrategy

def get_exploration_rate(game_number):
    return 0.1 * math.pow(1 - 0.000002, game_number)

def print_board(board_matrix):
    symbols = {1: 'X', -1: 'O', 0: '.'}
    for row in reversed(board_matrix):
        print('|' + ''.join(symbols[val] for val in row) + '|')
    print('+-------+\n')

def train_networkB_vs_mcts50(num_games=150000, batch_size=100):
    netB = NetworkB(learning_rate=1e-3)
    model_id = 'X'
    opponent_id = 'O'

    agent = NNStrategy(
        netB, 
        player_id=model_id,
        discount_factor=0.9,
        exploration_rate=0.2,
        batch_size=batch_size
    )

    opponent = MctsStrategy(rollout_limit=150)

    # Arrays to store metrics per block
    win_rates = []
    draw_rates = []
    first_win_rates = []
    second_win_rates = []
    x_vals = []

    # Block counters
    block_size = 500
    total_wins = 0
    draws = 0
    model_first_games = 0
    model_second_games = 0
    model_first_wins = 0
    model_second_wins = 0

    # For sample games
    sample_win_game = []
    sample_loss_game = []

    for game_num in trange(num_games, desc="Training"):
        game = Game()
        game.current_player = random.choice(['X', 'O'])
        agent.exploration_rate = get_exploration_rate(game_num)

        snapshots = []
        model_moved_first = (game.current_player == model_id)

        while not game.finished:
            snapshots.append(agent.encode_board_relative(game.board.clone()))
            if game.current_player == model_id:
                agent.record_state(game.clone())
                move = agent.pick_move(game)
            else:
                move = opponent.move(game.clone(), opponent_id)
            game.move(move)

        # Who won?
        agent.game_over(
            game.winner,
            last_player=opponent_id if game.current_player == model_id else model_id
        )

        if game.winner == model_id:
            total_wins += 1
            if model_moved_first:
                model_first_wins += 1
                if not sample_win_game:
                    sample_win_game = snapshots
            else:
                model_second_wins += 1
                if not sample_win_game:
                    sample_win_game = snapshots
        elif game.winner is None:
            draws += 1
            if not sample_loss_game:
                sample_loss_game = snapshots
        else:
            # The model lost
            if not sample_loss_game:
                sample_loss_game = snapshots

        # Increment how many times model went first/second
        if model_moved_first:
            model_first_games += 1
        else:
            model_second_games += 1

        # After each block of 250 games, compute stats
        if (game_num + 1) % block_size == 0:
            block_win_rate = total_wins / block_size
            block_draw_rate = draws / block_size

            # If model_first_games or model_second_games is zero, avoid division by zero
            block_first_win_rate = (model_first_wins / model_first_games) if model_first_games > 0 else 0
            block_second_win_rate = (model_second_wins / model_second_games) if model_second_games > 0 else 0

            win_rates.append(block_win_rate)
            draw_rates.append(block_draw_rate)
            first_win_rates.append(block_first_win_rate)
            second_win_rates.append(block_second_win_rate)
            x_vals.append(game_num + 1)

            print(f"\n--- Block {game_num + 1} ---")
            print(f"  Total wins (out of {block_size}):  {total_wins} ({block_win_rate:.2%})")
            print(f"  Draws (out of {block_size}):       {draws} ({block_draw_rate:.2%})")
            print(f"  Model went first in {model_first_games} games; won {model_first_wins} "
                  f"({block_first_win_rate:.2%} of those first-player games)")
            print(f"  Model went second in {model_second_games} games; won {model_second_wins} "
                  f"({block_second_win_rate:.2%} of those second-player games)")

            # Reset block counters
            total_wins = 0
            draws = 0
            model_first_games = 0
            model_second_games = 0
            model_first_wins = 0
            model_second_wins = 0

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, win_rates, label="ModelB Win Rate", linewidth=2)
    plt.plot(x_vals, draw_rates, label="Draw Rate", linewidth=2)
    plt.plot(x_vals, first_win_rates, label="First Player Win Rate", linewidth=2)
    plt.plot(x_vals, second_win_rates, label="Second Player Win Rate", linewidth=2)
    plt.xlabel("Games Played")
    plt.ylabel("Rate (per 500-game block)")
    plt.title("NetworkB vs MCTS150 Bot")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("win_rates_networkB.png")
    plt.show()

    print("\n🔍 Sample game the model WON")
    if sample_win_game:
        print_board(sample_win_game[-1])
    else:
        print("No sample win game recorded.")

    print("\n🔍 Sample game the model LOST")
    if sample_loss_game:
        print_board(sample_loss_game[-1])
    else:
        print("No sample loss game recorded.")
    return netB

if __name__ == "__main__":
    trained_model = train_networkB_vs_mcts50()
    trained_model.save("networkB_mcts50.h5")