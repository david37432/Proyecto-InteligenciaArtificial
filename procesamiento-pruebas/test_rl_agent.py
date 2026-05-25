"""
test_rl_agent.py
Enfrenta QLearningAgent contra un agente aleatorio en N partidas
e imprime la tasa de victorias, derrotas y empates.
"""

import random
import numpy as np
from rl_agent import QLearningAgent

ROWS = 6
COLS = 7


def random_agent_act(board):
    free = [c for c in range(COLS) if board[0, c] == 0]
    return random.choice(free) if free else 3


def apply_move(board, col, player):
    new_board = board.copy()
    for r in range(ROWS - 1, -1, -1):
        if new_board[r, col] == 0:
            new_board[r, col] = player
            break
    return new_board


def check_winner(board):
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r, c]
            if p == 0:
                continue
            if c + 3 < COLS and all(board[r, c+i] == p for i in range(4)):
                return p
            if r + 3 < ROWS and all(board[r+i, c] == p for i in range(4)):
                return p
            if r + 3 < ROWS and c + 3 < COLS and all(board[r+i, c+i] == p for i in range(4)):
                return p
            if r + 3 < ROWS and c - 3 >= 0 and all(board[r+i, c-i] == p for i in range(4)):
                return p
    return 0


def play_game(rl_agent, rl_is_player1=True):
    """
    Juega una partida completa.
    rl_is_player1=True  → RL juega como -1 (mueve primero)
    rl_is_player1=False → RL juega como +1 (mueve segundo)
    Devuelve 'win', 'loss' o 'draw' desde la perspectiva del RL.
    """
    board = np.zeros((ROWS, COLS), dtype=int)
    rl_player = -1 if rl_is_player1 else 1
    current_player = -1  # siempre empieza -1

    while True:
        free_cols = [c for c in range(COLS) if board[0, c] == 0]
        if not free_cols:
            return 'draw'

        if current_player == rl_player:
            col = rl_agent.act(board)
        else:
            col = random_agent_act(board)

        board = apply_move(board, col, current_player)
        winner = check_winner(board)

        if winner != 0:
            if winner == rl_player:
                return 'win'
            else:
                return 'loss'

        current_player = -current_player


def run_tournament(n_episodes=1000, n_games=100):
    print(f"Entrenando QLearningAgent con {n_episodes} episodios...")
    agent = QLearningAgent(player=-1, n_episodes=n_episodes)
    agent.mount()
    print("Entrenamiento completado.")
    print(f"Pesos aprendidos: {agent.weights}")

    wins = draws = losses = 0
    for i in range(n_games):
        rl_is_p1 = (i % 2 == 0)
        result = play_game(agent, rl_is_player1=rl_is_p1)
        if result == 'win':
            wins += 1
        elif result == 'draw':
            draws += 1
        else:
            losses += 1

    print(f"\n=== Resultados ({n_games} partidas) ===")
    print(f"  Victorias: {wins}  ({100*wins/n_games:.1f}%)")
    print(f"  Empates:   {draws}  ({100*draws/n_games:.1f}%)")
    print(f"  Derrotas:  {losses}  ({100*losses/n_games:.1f}%)")
    return wins, draws, losses


if __name__ == "__main__":
    run_tournament(n_episodes=1000, n_games=100)
