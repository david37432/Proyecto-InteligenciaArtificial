# rl_agent.py
# Agente basado en Q-Learning con auto-juego (Cross-MDP Policy Iteration)
# Fundamento teórico: Diapositiva 12 - Competitive MDPs, curso FIA 2026.1

import numpy as np
import random
import pickle
import os
from connect4.policy import Policy
from connect4.connect_state import ConnectState

ROWS = 6
COLS = 7
CENTER_COL = 3


class QLearningAgent(Policy):
    """
    Agente de Q-Learning con auto-juego basado en el marco de
    Competitive MDPs (Alternating Markov Games) visto en clase.

    Fase offline: aprende jugando contra sí mismo (Cross-MDP self-play).
    Fase online: actúa greedily usando los pesos aprendidos.

    La Q-function se aproxima con una combinación lineal de features:
      Q(s, a) ≈ w · φ(s, a)
    """

    def __init__(self,
                 player: int = -1,
                 alpha: float = 0.01,
                 gamma: float = 1.0,
                 epsilon: float = 0.2,
                 n_episodes: int = 5000,
                 weights_file: str = None):
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.weights_file = weights_file
        self.weights = None

    def mount(self, *args, **kwargs) -> None:
        n_features = 8

        if self.weights_file and os.path.exists(self.weights_file):
            with open(self.weights_file, 'rb') as f:
                self.weights = pickle.load(f)
        else:
            self.weights = np.zeros(n_features)
            self._train()
            if self.weights_file:
                with open(self.weights_file, 'wb') as f:
                    pickle.dump(self.weights, f)

    def act(self, s: np.ndarray) -> int:
        free_cols = [c for c in range(COLS) if s[0, c] == 0]
        if not free_cols:
            return 3

        # Detectar victoria inmediata
        for col in free_cols:
            if self._is_winning_move(s, col, self.player):
                return col

        # Detectar bloqueo inmediato
        for col in free_cols:
            if self._is_winning_move(s, col, -self.player):
                return col

        # Greedy sobre Q-values aprendidos
        if self.weights is not None:
            q_values = [self._q_value(s, col, self.player) for col in free_cols]
            best_col = free_cols[np.argmax(q_values)]
            return best_col

        return random.choice(free_cols)

    # ------------------------------------------------------------------ #
    #  ENTRENAMIENTO (Cross-MDP Self-Play)
    # ------------------------------------------------------------------ #

    def _train(self):
        """
        Entrena mediante auto-juego (Cross-MDP Policy Iteration).
        El agente juega ambos roles alternando y usa el truco bipolar:
        al cambiar de turno, niega el valor futuro (×−γ).
        """
        for _ in range(self.n_episodes):
            self._run_episode()

    def _run_episode(self):
        board = np.zeros((ROWS, COLS), dtype=int)
        current_player = -1
        history = []  # (board_before, col, player)

        while True:
            free_cols = [c for c in range(COLS) if board[0, c] == 0]
            if not free_cols:
                self._update_weights_from_history(history, reward=0.0)
                return

            if random.random() < self.epsilon:
                col = random.choice(free_cols)
            else:
                q_vals = [self._q_value(board, c, current_player) for c in free_cols]
                col = free_cols[np.argmax(q_vals)]

            history.append((board.copy(), col, current_player))
            board = self._apply_move(board, col, current_player)

            winner = self._check_winner(board)
            if winner != 0:
                self._update_weights_from_history(history, reward=1.0)
                return

            current_player = -current_player

    def _update_weights_from_history(self, history, reward: float):
        """
        Propaga la recompensa hacia atrás con el truco bipolar del curso:
        cada cambio de turno niega la recompensa (× −γ).
        """
        r = reward
        for board, col, player in reversed(history):
            phi = self._extract_features(board, col, player)
            q_current = np.dot(self.weights, phi)
            error = r - q_current
            self.weights += self.alpha * error * phi
            r = -self.gamma * r  # truco bipolar

    # ------------------------------------------------------------------ #
    #  FEATURES Y Q-VALUE
    # ------------------------------------------------------------------ #

    def _q_value(self, board: np.ndarray, col: int, player: int) -> float:
        phi = self._extract_features(board, col, player)
        return float(np.dot(self.weights, phi))

    def _extract_features(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
        """
        Vector de 8 features que describen la calidad del movimiento (col, player).

        0: victoria inmediata
        1: bloqueo inmediato del oponente
        2: preferencia de columna central (normalizada)
        3: amenazas propias creadas (3 fichas + 1 libre)
        4: amenazas del oponente bloqueadas
        5: ventanas con 2 fichas propias + 2 libres
        6: control de fila inferior
        7: bias constante
        """
        phi = np.zeros(8)
        row = self._get_drop_row(board, col)
        if row < 0:
            return phi

        test_board = board.copy()
        test_board[row, col] = player

        if self._check_winner(test_board) == player:
            phi[0] = 1.0
            phi[7] = 1.0
            return phi

        opponent = -player

        opp_board = board.copy()
        opp_board[row, col] = opponent
        if self._check_winner(opp_board) == opponent:
            phi[1] = 1.0

        phi[2] = (COLS - 1 - abs(col - CENTER_COL)) / CENTER_COL

        phi[3] = self._count_threats(test_board, player) / 10.0

        base_opp_threats = self._count_threats(board, opponent)
        new_opp_threats = self._count_threats(test_board, opponent)
        phi[4] = max(0, base_opp_threats - new_opp_threats) / 10.0

        phi[5] = self._count_two_in_row(test_board, player) / 20.0

        bottom_row = board[ROWS - 1, :]
        phi[6] = np.sum(bottom_row == player) / COLS

        phi[7] = 1.0

        return phi

    def _count_threats(self, board: np.ndarray, player: int) -> int:
        """Ventanas de 4 con exactamente 3 fichas del jugador y 1 libre."""
        count = 0
        for r in range(ROWS):
            for c in range(COLS - 3):
                window = board[r, c:c+4]
                if np.sum(window == player) == 3 and np.sum(window == 0) == 1:
                    count += 1
        for c in range(COLS):
            for r in range(ROWS - 3):
                window = board[r:r+4, c]
                if np.sum(window == player) == 3 and np.sum(window == 0) == 1:
                    count += 1
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                window = [board[r+i, c+i] for i in range(4)]
                if window.count(player) == 3 and window.count(0) == 1:
                    count += 1
        for r in range(ROWS - 3):
            for c in range(3, COLS):
                window = [board[r+i, c-i] for i in range(4)]
                if window.count(player) == 3 and window.count(0) == 1:
                    count += 1
        return count

    def _count_two_in_row(self, board: np.ndarray, player: int) -> int:
        """Ventanas de 4 con exactamente 2 fichas del jugador y 2 libres."""
        count = 0
        for r in range(ROWS):
            for c in range(COLS - 3):
                window = board[r, c:c+4]
                if np.sum(window == player) == 2 and np.sum(window == 0) == 2:
                    count += 1
        for c in range(COLS):
            for r in range(ROWS - 3):
                window = board[r:r+4, c]
                if np.sum(window == player) == 2 and np.sum(window == 0) == 2:
                    count += 1
        return count

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
                if c + 3 < COLS and all(board[r, c+i] == p for i in range(4)):
                    return p
                if r + 3 < ROWS and all(board[r+i, c] == p for i in range(4)):
                    return p
                if r + 3 < ROWS and c + 3 < COLS and all(board[r+i, c+i] == p for i in range(4)):
                    return p
                if r + 3 < ROWS and c - 3 >= 0 and all(board[r+i, c-i] == p for i in range(4)):
                    return p
        return 0
