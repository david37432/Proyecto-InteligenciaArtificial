# groups/Martin Jerez/policy.py
# QLearningAgent - Agente RL con Alpha-Beta Minimax
# Fundamento: Competitive MDPs (Diapo 12) + Online Policy Improvement (Diapo 13)

import numpy as np
import random
from connect4.policy import Policy

ROWS = 6
COLS = 7
CENTER_COL = 3
N_FEATURES = 8
INF = float('inf')
_CW = np.array([0, 1, 2, 3, 2, 1, 0], dtype=float)


class QLearningAgent(Policy):
    """
    Agente Q-Learning + Alpha-Beta Minimax para Connect-4.

    Fase offline: Q-learning con self-play (Cross-MDP) aprende pesos w para V(s)=w*psi(s).
    Fase online:  Alpha-Beta profundidad 4 usa V(s) como heuristica de hoja.
    """

    def __init__(self,
                 player: int = -1,
                 alpha: float = 0.01,
                 gamma: float = 1.0,
                 epsilon: float = 0.3,
                 epsilon_end: float = 0.05,
                 n_episodes: int = 5000,
                 search_depth: int = 4):
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon
        self.epsilon_end = epsilon_end
        self.n_episodes = n_episodes
        self.search_depth = search_depth
        self.weights = None

    def mount(self, *args, **kwargs) -> None:
        self.weights = np.zeros(N_FEATURES)
        self._train()

    # ------------------------------------------------------------------ #
    #  ACCION ONLINE
    # ------------------------------------------------------------------ #

    def act(self, s: np.ndarray) -> int:
        if self.weights is None:
            self.mount()

        free = [c for c in range(COLS) if s[0, c] == 0]
        if not free:
            return CENTER_COL

        for col in free:
            if self._is_winning_move(s, col, self.player):
                return col
        for col in free:
            if self._is_winning_move(s, col, -self.player):
                return col

        best_col = min(free, key=lambda c: abs(c - CENTER_COL))
        best_val = -INF
        alpha = -INF

        for col in self._order_moves(s, free, self.player):
            child = self._apply_move(s, col, self.player)
            val = self._alphabeta(child, self.search_depth - 1, alpha, INF, False)
            if val > best_val:
                best_val = val
                best_col = col
            alpha = max(alpha, val)
            if best_val > 9000:
                break

        return best_col

    # ------------------------------------------------------------------ #
    #  ALPHA-BETA MINIMAX
    # ------------------------------------------------------------------ #

    def _alphabeta(self, board, depth, alpha, beta, maximizing):
        winner = self._check_winner(board)
        if winner == self.player:
            return 10000 + depth
        if winner == -self.player:
            return -10000 - depth

        free = [c for c in range(COLS) if board[0, c] == 0]
        if not free:
            return 0.0
        if depth == 0:
            return float(np.dot(self.weights, self._board_features(board, self.player)))

        current_mover = self.player if maximizing else -self.player
        ordered = self._order_moves(board, free, current_mover)

        if maximizing:
            val = -INF
            for col in ordered:
                val = max(val, self._alphabeta(self._apply_move(board, col, current_mover),
                                               depth - 1, alpha, beta, False))
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            return val
        else:
            val = INF
            for col in ordered:
                val = min(val, self._alphabeta(self._apply_move(board, col, current_mover),
                                               depth - 1, alpha, beta, True))
                beta = min(beta, val)
                if alpha >= beta:
                    break
            return val

    def _order_moves(self, board, free, player):
        win_moves = [c for c in free if self._is_winning_move(board, c, player)]
        if win_moves:
            return win_moves + sorted([c for c in free if c not in win_moves],
                                      key=lambda c: abs(c - CENTER_COL))
        block_moves = [c for c in free if self._is_winning_move(board, c, -player)]
        rest = sorted([c for c in free if c not in block_moves],
                      key=lambda c: abs(c - CENTER_COL))
        return block_moves + rest

    # ------------------------------------------------------------------ #
    #  FEATURES (vectorizadas, 8 dimensiones)
    # ------------------------------------------------------------------ #

    def _board_features(self, board, player):
        opp = -player
        P = board == player
        O = board == opp
        E = board == 0
        th_p = tw_p = th_o = tw_o = 0

        for c in range(COLS - 3):
            wp = P[:, c].astype(int) + P[:, c+1] + P[:, c+2] + P[:, c+3]
            wo = O[:, c].astype(int) + O[:, c+1] + O[:, c+2] + O[:, c+3]
            we = E[:, c].astype(int) + E[:, c+1] + E[:, c+2] + E[:, c+3]
            th_p += int(((wp == 3) & (we == 1)).sum())
            th_o += int(((wo == 3) & (we == 1)).sum())
            tw_p += int(((wp == 2) & (we == 2)).sum())
            tw_o += int(((wo == 2) & (we == 2)).sum())

        for r in range(ROWS - 3):
            wp = P[r, :].astype(int) + P[r+1, :] + P[r+2, :] + P[r+3, :]
            wo = O[r, :].astype(int) + O[r+1, :] + O[r+2, :] + O[r+3, :]
            we = E[r, :].astype(int) + E[r+1, :] + E[r+2, :] + E[r+3, :]
            th_p += int(((wp == 3) & (we == 1)).sum())
            th_o += int(((wo == 3) & (we == 1)).sum())
            tw_p += int(((wp == 2) & (we == 2)).sum())
            tw_o += int(((wo == 2) & (we == 2)).sum())

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

        phi = np.empty(N_FEATURES)
        phi[0] = th_p / 10.0
        phi[1] = th_o / 10.0
        phi[2] = tw_p / 20.0
        phi[3] = tw_o / 20.0
        phi[4] = float(_CW @ P.sum(axis=0).astype(float)) / 42.0
        phi[5] = float(_CW @ O.sum(axis=0).astype(float)) / 42.0
        phi[6] = float(P[ROWS - 1, :].sum()) / COLS
        phi[7] = 1.0
        return phi

    # ------------------------------------------------------------------ #
    #  ENTRENAMIENTO
    # ------------------------------------------------------------------ #

    def _train(self):
        for ep in range(self.n_episodes):
            t = ep / max(1, self.n_episodes - 1)
            epsilon = self.epsilon_start * (1 - t) + self.epsilon_end * t
            self._run_episode(epsilon)

    def _run_episode(self, epsilon):
        board = np.zeros((ROWS, COLS), dtype=int)
        current_player = -1
        history = []

        while True:
            free = [c for c in range(COLS) if board[0, c] == 0]
            if not free:
                self._update_weights(history, 0.0)
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

            if self._check_winner(board) != 0:
                self._update_weights(history, 1.0)
                return
            current_player = -current_player

    def _update_weights(self, history, reward):
        r = reward
        for board, col, player in reversed(history):
            new_board = self._apply_move(board, col, player)
            phi = self._board_features(new_board, player)
            q_current = float(np.dot(self.weights, phi))
            self.weights += self.alpha * (r - q_current) * phi
            r = -self.gamma * r

    # ------------------------------------------------------------------ #
    #  UTILIDADES
    # ------------------------------------------------------------------ #

    def _get_drop_row(self, board, col):
        for r in range(ROWS - 1, -1, -1):
            if board[r, col] == 0:
                return r
        return -1

    def _apply_move(self, board, col, player):
        new_board = board.copy()
        row = self._get_drop_row(new_board, col)
        if row >= 0:
            new_board[row, col] = player
        return new_board

    def _is_winning_move(self, board, col, player):
        row = self._get_drop_row(board, col)
        if row < 0:
            return False
        test = board.copy()
        test[row, col] = player
        return self._check_winner(test) == player

    def _check_winner(self, board):
        for r in range(ROWS):
            for c in range(COLS):
                p = board[r, c]
                if p == 0:
                    continue
                if c+3<COLS and board[r,c+1]==p and board[r,c+2]==p and board[r,c+3]==p:
                    return p
                if r+3<ROWS and board[r+1,c]==p and board[r+2,c]==p and board[r+3,c]==p:
                    return p
                if r+3<ROWS and c+3<COLS \
                        and board[r+1,c+1]==p and board[r+2,c+2]==p and board[r+3,c+3]==p:
                    return p
                if r+3<ROWS and c-3>=0 \
                        and board[r+1,c-1]==p and board[r+2,c-2]==p and board[r+3,c-3]==p:
                    return p
        return 0
