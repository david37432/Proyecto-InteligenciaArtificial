# rl_agent.py
# Agente Q-Learning + Alpha-Beta Minimax (Cross-MDP Policy Iteration)
# Fundamento teorico: Diapositiva 12 - Competitive MDPs, curso FIA 2026.1
#
# La funcion de evaluacion V(s) = w * psi(s) se aprende offline via Q-learning.
# Durante el juego, Alpha-Beta explota esa funcion buscando 4 niveles profundo,
# detectando tacticas (forks, dobles amenazas) que el lookup plano no podia ver.

import numpy as np
import random
import pickle
import os
from connect4.policy import Policy

ROWS = 6
COLS = 7
CENTER_COL = 3
N_FEATURES = 8
INF = float('inf')

# Pesos de columna: centro vale mas
_CW = np.array([0, 1, 2, 3, 2, 1, 0], dtype=float)


class QLearningAgent(Policy):
    """
    Agente de Q-Learning con funcion de evaluacion aprendida + Alpha-Beta Minimax.

    Fase offline: Q-learning con self-play (Cross-MDP) aprende pesos w para V(s)=w*psi(s).
    Fase online:  Alpha-Beta profundidad 4 usa V(s) como heuristica de hoja.

    Diferencia clave respecto a MCTS: la evaluacion es APRENDIDA, no aleatoria.
    """

def __init__(self,
             player: int = -1,
             alpha: float = 0.01,
             gamma: float = 1.0,
             epsilon: float = 0.3,       # epsilon inicial (decae durante entrenamiento)
             epsilon_end: float = 0.05,   # epsilon final
             n_episodes: int = 5000,
             search_depth: int = 4,       # profundidad alpha-beta
             weights_file: str = None):
    self.player = player
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon_start = epsilon
    self.epsilon_end = epsilon_end
    self.n_episodes = n_episodes
    self.search_depth = search_depth
    self.weights_file = weights_file
    self.weights = None

def mount(self, *args, **kwargs) -> None:
    if self.weights_file and os.path.exists(self.weights_file):
        with open(self.weights_file, 'rb') as f:
            self.weights = pickle.load(f)
    else:
        self.weights = np.zeros(N_FEATURES)
        self._train()
        if self.weights_file:
            with open(self.weights_file, 'wb') as f:
                pickle.dump(self.weights, f)

# ------------------------------------------------------------------ #
#  ACCION ONLINE (Alpha-Beta + evaluacion aprendida)
# ------------------------------------------------------------------ #

def act(self, s: np.ndarray) -> int:
    free = [c for c in range(COLS) if s[0, c] == 0]
    if not free:
        return CENTER_COL

    # Ganar inmediatamente si es posible
    for col in free:
        if self._is_winning_move(s, col, self.player):
            return col

    # Bloquear victoria inmediata del oponente
    for col in free:
        if self._is_winning_move(s, col, -self.player):
            return col

    # Alpha-Beta con funcion de evaluacion aprendida
    best_col = min(free, key=lambda c: abs(c - CENTER_COL))
    best_val = -INF
    alpha = -INF

    for col in self._order_moves(s, free, self.player):
        child = self._apply_move(s, col, self.player)
        val = self._alphabeta(child, self.search_depth - 1, alpha, INF, maximizing=False)
        if val > best_val:
            best_val = val
            best_col = col
        alpha = max(alpha, val)
        if best_val > 9000:   # victoria forzada encontrada
            break

    return best_col

# ------------------------------------------------------------------ #
#  ALPHA-BETA MINIMAX
# ------------------------------------------------------------------ #

def _alphabeta(self, board: np.ndarray, depth: int,
               alpha: float, beta: float, maximizing: bool) -> float:
    winner = self._check_winner(board)
    if winner == self.player:
        return 10000 + depth      # ganar antes es mejor
    if winner == -self.player:
        return -10000 - depth     # perder antes es peor

    free = [c for c in range(COLS) if board[0, c] == 0]
    if not free:
        return 0.0                # empate

    if depth == 0:
        return self._evaluate_board(board)

    current_mover = self.player if maximizing else -self.player
    ordered = self._order_moves(board, free, current_mover)

    if maximizing:
        val = -INF
        for col in ordered:
            child = self._apply_move(board, col, current_mover)
            val = max(val, self._alphabeta(child, depth - 1, alpha, beta, False))
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return val
    else:
        val = INF
        for col in ordered:
            child = self._apply_move(board, col, current_mover)
            val = min(val, self._alphabeta(child, depth - 1, alpha, beta, True))
            beta = min(beta, val)
            if alpha >= beta:
                break
        return val

def _order_moves(self, board: np.ndarray, free: list, player: int) -> list:
    """
    Ordena movimientos para maximizar la poda alpha-beta:
    ganadores primero → bloqueos → centro hacia afuera.
    """
    win_moves = [c for c in free if self._is_winning_move(board, c, player)]
    if win_moves:
        others = [c for c in free if c not in win_moves]
        return win_moves + sorted(others, key=lambda c: abs(c - CENTER_COL))

    block_moves = [c for c in free if self._is_winning_move(board, c, -player)]
    rest = [c for c in free if c not in block_moves]
    return block_moves + sorted(rest, key=lambda c: abs(c - CENTER_COL))

def _evaluate_board(self, board: np.ndarray) -> float:
    """V(s) = w * psi(s, self.player) — heuristica de hoja aprendida."""
    return float(np.dot(self.weights, self._board_features(board, self.player)))

# ------------------------------------------------------------------ #
#  FEATURES DE TABLERO (vectorizadas, 8 dimensiones)
# ------------------------------------------------------------------ #

def _board_features(self, board: np.ndarray, player: int) -> np.ndarray:
    """
    8 features rapidas (vectorizadas) desde la perspectiva de `player`.

      0: amenazas propias   (ventanas 3 fichas + 1 libre)
      1: amenazas oponente  (ventanas 3 fichas + 1 libre)
      2: dobles propias     (ventanas 2 fichas + 2 libres)
      3: dobles oponente
      4: control columnas centrales (ponderado, propio)
      5: control columnas centrales (ponderado, oponente)
      6: piezas en fila inferior propia
      7: bias constante
    """
    opp = -player
    P = board == player   # bool (ROWS, COLS)
    O = board == opp
    E = board == 0

    th_p = tw_p = th_o = tw_o = 0

    # --- Horizontal (4 ventanas de longitud por fila, vectorizadas) ---
    for c in range(COLS - 3):
        wp = P[:, c].astype(int) + P[:, c+1] + P[:, c+2] + P[:, c+3]
        wo = O[:, c].astype(int) + O[:, c+1] + O[:, c+2] + O[:, c+3]
        we = E[:, c].astype(int) + E[:, c+1] + E[:, c+2] + E[:, c+3]
        th_p += int(((wp == 3) & (we == 1)).sum())
        th_o += int(((wo == 3) & (we == 1)).sum())
        tw_p += int(((wp == 2) & (we == 2)).sum())
        tw_o += int(((wo == 2) & (we == 2)).sum())

    # --- Vertical (3 ventanas de altura por columna, vectorizadas) ---
    for r in range(ROWS - 3):
        wp = P[r, :].astype(int) + P[r+1, :] + P[r+2, :] + P[r+3, :]
        wo = O[r, :].astype(int) + O[r+1, :] + O[r+2, :] + O[r+3, :]
        we = E[r, :].astype(int) + E[r+1, :] + E[r+2, :] + E[r+3, :]
        th_p += int(((wp == 3) & (we == 1)).sum())
        th_o += int(((wo == 3) & (we == 1)).sum())
        tw_p += int(((wp == 2) & (we == 2)).sum())
        tw_o += int(((wo == 2) & (we == 2)).sum())

    # --- Diagonales (12 + 12 = 24 iteraciones escalares) ---
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            p = int(P[r,c])+int(P[r+1,c+1])+int(P[r+2,c+2])+int(P[r+3,c+3])
            o = int(O[r,c])+int(O[r+1,c+1])+int(O[r+2,c+2])+int(O[r+3,c+3])
            e = int(E[r,c])+int(E[r+1,c+1])+int(E[r+2,c+2])+int(E[r+3,c+3])
            if p == 3 and e == 1: th_p += 1
            if o == 3 and e == 1: th_o += 1
            if p == 2 and e == 2: tw_p += 1
            if o == 2 and e == 2: tw_o += 1

    for r in range(ROWS - 3):
        for c in range(3, COLS):
            p = int(P[r,c])+int(P[r+1,c-1])+int(P[r+2,c-2])+int(P[r+3,c-3])
            o = int(O[r,c])+int(O[r+1,c-1])+int(O[r+2,c-2])+int(O[r+3,c-3])
            e = int(E[r,c])+int(E[r+1,c-1])+int(E[r+2,c-2])+int(E[r+3,c-3])
            if p == 3 and e == 1: th_p += 1
            if o == 3 and e == 1: th_o += 1
            if p == 2 and e == 2: tw_p += 1
            if o == 2 and e == 2: tw_o += 1

    col_sums_p = P.sum(axis=0).astype(float)
    col_sums_o = O.sum(axis=0).astype(float)

    phi = np.empty(N_FEATURES)
    phi[0] = th_p / 10.0
    phi[1] = th_o / 10.0
    phi[2] = tw_p / 20.0
    phi[3] = tw_o / 20.0
    phi[4] = float(_CW @ col_sums_p) / 42.0
    phi[5] = float(_CW @ col_sums_o) / 42.0
    phi[6] = float(P[ROWS - 1, :].sum()) / COLS
    phi[7] = 1.0

    return phi

# ------------------------------------------------------------------ #
#  ENTRENAMIENTO (Cross-MDP Self-Play con epsilon decreciente)
# ------------------------------------------------------------------ #

def _train(self):
    for ep in range(self.n_episodes):
        t = ep / max(1, self.n_episodes - 1)
        epsilon = self.epsilon_start * (1 - t) + self.epsilon_end * t
        self._run_episode(epsilon)

def _run_episode(self, epsilon: float):
    board = np.zeros((ROWS, COLS), dtype=int)
    current_player = -1
    history = []

    while True:
        free = [c for c in range(COLS) if board[0, c] == 0]
        if not free:
            self._update_weights_from_history(history, reward=0.0)
            return

        if random.random() < epsilon:
            col = random.choice(free)
        else:
            best_col, best_q = free[0], -INF
            for c in free:
                new_b = self._apply_move(board, c, current_player)
                q = float(np.dot(self.weights,
                                 self._board_features(new_b, current_player)))
                if q > best_q:
                    best_q = q
                    best_col = c
            col = best_col

        history.append((board.copy(), col, current_player))
        board = self._apply_move(board, col, current_player)

        winner = self._check_winner(board)
        if winner != 0:
            self._update_weights_from_history(history, reward=1.0)
            return

        current_player = -current_player

def _update_weights_from_history(self, history: list, reward: float):
    """
    Propaga la recompensa hacia atras con el truco bipolar del curso:
    al cambiar de turno, el valor se niega (x -gamma).
    """
    r = reward
    for board, col, player in reversed(history):
        new_board = self._apply_move(board, col, player)
        phi = self._board_features(new_board, player)
        q_current = float(np.dot(self.weights, phi))
        self.weights += self.alpha * (r - q_current) * phi
        r = -self.gamma * r   # truco bipolar

# ------------------------------------------------------------------ #
#  UTILIDADES
# ------------------------------------------------------------------ #

def _get_drop_row(self, board: np.ndarray, col: int) -> int:
    for r in range(ROWS - 1, -1, -1):
        if board[r, col] == 0:
            return r
    return -1

def _apply_move(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
    new_board = board.copy()
    row = self._get_drop_row(new_board, col)
    if row >= 0:
        new_board[row, col] = player
    return new_board

def _is_winning_move(self, board: np.ndarray, col: int, player: int) -> bool:
    row = self._get_drop_row(board, col)
    if row < 0:
        return False
    test = board.copy()
    test[row, col] = player
    return self._check_winner(test) == player

def _check_winner(self, board: np.ndarray) -> int:
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r, c]
            if p == 0:
                continue
            if c+3 < COLS and board[r,c+1]==p and board[r,c+2]==p and board[r,c+3]==p:
                return p
            if r+3 < ROWS and board[r+1,c]==p and board[r+2,c]==p and board[r+3,c]==p:
                return p
            if r+3 < ROWS and c+3 < COLS \
                    and board[r+1,c+1]==p and board[r+2,c+2]==p and board[r+3,c+3]==p:
                return p
            if r+3 < ROWS and c-3 >= 0 \
                    and board[r+1,c-1]==p and board[r+2,c-2]==p and board[r+3,c-3]==p:
                return p
    return 0
