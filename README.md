Connect‑X Reinforcement Learning Agents

This repository provides implementations of reinforcement‑learning agents for Connect‑4 and Connect‑5 games, leveraging neural networks and Monte Carlo Tree Search (MCTS). It includes board and game logic, two CNN architectures (NetworkA and NetworkB), a neural‑network strategy (NNStrategy), an MCTS opponent, and training scripts that pit the learned agent against the MCTS baseline.

📂 Repository Structure
├── Board.py             # Connect‑4 board logic (6×7)
├── Game.py              # Game engine with turn management and win detection
├── MCTS.py              # Monte Carlo Tree Search implementation (MctsStrategy)
├── NNStrategy.py        # Temporal‑difference‑style training strategy
├── NetworkA.py          # CNN #1 for value estimation (smaller filters)
├── NetworkB.py          # CNN #2 for value estimation (larger filters)
├── networkA_trials.py   # Train NetworkA vs MCTS on Connect‑4
├── networkB_trials.py   # Train NetworkB vs MCTS on Connect‑4
└── connect5.py          # Train NetworkB vs MCTS on Connect‑5 (8×9 board)

Requirements & Installation

Python 3.7+
TensorFlow
NumPy
Matplotlib
tqdm

1. Training Connect‑4 Agents
NetworkA (2×2 conv filters): python networkA_trials.py
NetworkB (4×4 conv filters): python networkB_trials.py

Both scripts print per‑block statistics (wins, draws, first/second‑player win rates) and save a plot (win_rates_networkA.png / win_rates_networkB.png).

2. Training Connect‑5 Agent
On an 8×9 board with 5×5 filters in NetworkB: python connect5.py
The script reports the same block‑level metrics and outputs connect5_stats.png.

Outputs
Model checkpoints: .h5 files saved by NetworkA.save() or NetworkB.save() (e.g., networkA.h5, networkB_connect5.h5).
Training curves: PNG (or switch to SVG/PDF via plt.savefig(..., format="svg")) showing win/draw rates.
Console logs: Per‑block summaries of performance metrics.

Customization
Board dimensions: Modify Board.rows, Board.cols, and the check_win threshold in Board.py (or connect5.py).
Conv filters: Adjust kernel sizes in NetworkA.py / NetworkB.py.
Training hyperparameters: Tweak num_games, batch_size, rollout_limit, or the exploration schedule in each script.
