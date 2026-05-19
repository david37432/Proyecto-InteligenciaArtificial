# groups/Martin Jerez/policy_hardcoded.py
# QLearningAgent con pesos pre-entrenados (hardcoded).
# act() es siempre instantaneo — sin entrenamiento en runtime.

import numpy as np
from connect4.policy import Policy

ROWS = 6
COLS = 7
CENTER_COL = 3
N_FEATURES = 8
INF = float('inf')
_CW = np.array([0, 1, 2, 3, 2, 1, 0], dtype=float)

# Pesos entrenados con 5000 episodios de Cross-MDP self-play
_PRETRAINED_WEIGHTS = np.array([
    0.7668944459511811,
    -0.8694685051723307,
    0.053516383877476065,
    -0.2671356659415824,
    0.21677403246978277,
    0.02008998179779194,
    -0.10724775702571433,
    -0.022472294485341063,
])


class QLearningAgent(Policy):
    """
    Agente Q-Learning + Alpha-Beta Minimax para Connect-4.

    Pesos pre-entrenados via Cross-MDP self-play (5000 episodios).
    act() responde en ~34ms — sin entrenamiento en runtime.
    """

    def __init__(self, player: int = -1, search_depth: int = 4, **kwargs):
        self.player = player
        self.search_depth = search_depth
        self.weights = _PRETRAINED_WEIGHTS.copy()

    def mount(self, *args, **kwargs) -> None:
        self.weights = _PRETRAINED_WEIGHTS.copy()

    # ------------------------------------------------------------------ #
    #  ACCION ONLINE
    # ------------------------------------------------------------------ #

    def act(self, s: np.ndarray) -> int:
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
    #  FEATURES (8 dimensiones)
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
