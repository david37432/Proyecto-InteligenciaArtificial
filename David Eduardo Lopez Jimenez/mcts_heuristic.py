# groups/David Eduardo Lopez Jimenez/mcts_heuristic.py

import numpy as np
import math
import random
from connect4.policy import Policy
from connect4.connect_state import ConnectState


class MCTSNode:
    """Nodo del árbol de Monte Carlo."""
    def __init__(self, state: ConnectState, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children: list['MCTSNode'] = []
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = state.get_free_cols()
        random.shuffle(self.untried_moves)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.41):
        best_value = -float('inf')
        best_child = None
        for child in self.children:
            if child.visits == 0:
                return child
            exploitation = child.wins / child.visits
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            ucb = exploitation + exploration
            if ucb > best_value:
                best_value = ucb
                best_child = child
        return best_child

    def add_child(self, move, child_state):
        child = MCTSNode(child_state, parent=self, move=move)
        self.children.append(child)
        return child


class MCTSAgentHeuristic(Policy):
    """MCTS con simulaciones guiadas por heurística optimizada."""
    def __init__(self, player: int = -1, num_simulations: int = 200):
        self.player = player
        self.num_simulations = num_simulations

    def mount(self) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        current_state = ConnectState(board=s, player=self.player)
        root = MCTSNode(current_state)

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not node.state.is_final():
                node = self._expand(node)
            result = self._simulate(node)
            self._backpropagate(node, result)

        if not root.children:
            return 3
        best_move = max(root.children, key=lambda c: c.visits).move
        return best_move

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.state.is_final() and node.is_fully_expanded():
            node = node.best_child()
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        move = node.untried_moves.pop()
        new_state = node.state.transition(move)
        return node.add_child(move, new_state)

    def _simulate(self, node: MCTSNode) -> float:
        state = node.state
        if state.is_final():
            return self._result_from_perspective(state, node.parent.state.player)

        # Simular usando el tablero como numpy array (mucho más rápido)
        current = state.board.copy()
        player = state.player
        while True:
            winner = self._get_winner(current)
            if winner != 0 or len([c for c in range(7) if current[0, c] == 0]) == 0:
                break
            move = self._heuristic_move(current, player)
            current = self._apply_move(current, move, player)
            player = -player

        final_winner = self._get_winner(current)
        if final_winner == 0:
            return 0.5
        return 1.0 if final_winner == node.parent.state.player else 0.0

    # --------------------------------------------------------
    # Funciones heurísticas optimizadas (solo numpy)
    # --------------------------------------------------------
    def _heuristic_move(self, board: np.ndarray, player: int) -> int:
        rows, cols = board.shape
        free_cols = [c for c in range(cols) if board[0, c] == 0]

        # 1. Victoria inmediata
        for col in free_cols:
            row = np.where(board[:, col] == 0)[0][-1]
            if self._check_win_at(board, row, col, player):
                return col

        # 2. Bloquear amenaza del oponente
        opponent = -player
        for col in free_cols:
            row = np.where(board[:, col] == 0)[0][-1]
            if self._check_win_at(board, row, col, opponent):
                return col

        # 3. Preferir centro con pesos aleatorios
        weights = {0:1, 1:2, 2:3, 3:4, 4:3, 5:2, 6:1}
        col_weights = [weights[c] for c in free_cols]
        total = sum(col_weights)
        probs = [w/total for w in col_weights]
        return np.random.choice(free_cols, p=probs)

    def _check_win_at(self, board: np.ndarray, r: int, c: int, player: int) -> bool:
        """Verifica si colocar una ficha en (r,c) produce 4 en línea para 'player'."""
        rows, cols = board.shape

        # Horizontal
        count = 1
        for dc in (-1, 1):
            nc = c + dc
            while 0 <= nc < cols and board[r, nc] == player:
                count += 1
                nc += dc
        if count >= 4: return True

        # Vertical (solo hacia abajo)
        count = 1
        nr = r + 1
        while nr < rows and board[nr, c] == player:
            count += 1
            nr += 1
        if count >= 4: return True

        # Diagonal principal (\)
        count = 1
        for dr, dc in ((-1,-1), (1,1)):
            nr, nc = r + dr, c + dc
            while 0 <= nr < rows and 0 <= nc < cols and board[nr, nc] == player:
                count += 1
                nr += dr
                nc += dc
        if count >= 4: return True

        # Diagonal secundaria (/)
        count = 1
        for dr, dc in ((-1,1), (1,-1)):
            nr, nc = r + dr, c + dc
            while 0 <= nr < rows and 0 <= nc < cols and board[nr, nc] == player:
                count += 1
                nr += dr
                nc += dc
        if count >= 4: return True

        return False

    def _apply_move(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
        """Devuelve una copia del tablero tras colocar la ficha en la columna."""
        new_board = board.copy()
        row = np.where(new_board[:, col] == 0)[0][-1]
        new_board[row, col] = player
        return new_board

    def _get_winner(self, board: np.ndarray) -> int:
        """Determina si hay un ganador en el tablero (rápido con numpy)."""
        rows, cols = board.shape
        for r in range(rows):
            for c in range(cols):
                player = board[r, c]
                if player == 0:
                    continue
                # Derecha
                if c + 3 < cols and all(board[r, c+i] == player for i in range(4)):
                    return player
                # Abajo
                if r + 3 < rows and all(board[r+i, c] == player for i in range(4)):
                    return player
                # Diagonal derecha-abajo
                if r + 3 < rows and c + 3 < cols and all(board[r+i, c+i] == player for i in range(4)):
                    return player
                # Diagonal izquierda-abajo
                if r + 3 < rows and c - 3 >= 0 and all(board[r+i, c-i] == player for i in range(4)):
                    return player
        return 0

    # --------------------------------------------------------
    # Retropropagación y resultado (sin cambios)
    # --------------------------------------------------------
    def _backpropagate(self, node: MCTSNode, result: float):
        while node is not None:
            node.visits += 1
            node.wins += result
            result = 1.0 - result if result != 0.5 else 0.5
            node = node.parent

    def _result_from_perspective(self, final_state: ConnectState, perspective_player: int) -> float:
        winner = final_state.get_winner()
        if winner == 0:
            return 0.5
        return 1.0 if winner == perspective_player else 0.0