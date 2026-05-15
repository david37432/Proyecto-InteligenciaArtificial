import numpy as np
import math
import random
from connect4.policy import Policy
from connect4.connect_state import ConnectState

class MCTSNode:
    def __init__(self, state: ConnectState, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
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


class MCTSAgentHeuristic:
    def __init__(self, player: int = -1, num_simulations: int = 600):
        self.player = player
        self.num_simulations = num_simulations

    def mount(self, *args, **kwargs):
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
            free = [c for c in range(7) if s[0, c] == 0]
            return random.choice(free) if free else 3
        return max(root.children, key=lambda c: c.visits).move

    def _select(self, node):
        while not node.state.is_final() and node.is_fully_expanded():
            node = node.best_child()
        return node

    def _expand(self, node):
        move = node.untried_moves.pop()
        new_state = node.state.transition(move)
        return node.add_child(move, new_state)

    def _simulate(self, node):
        state = node.state
        if state.is_final():
            perspective = node.parent.state.player if node.parent else state.player
            return self._result_from_perspective(state, perspective)

        board = state.board.copy()
        player = state.player
        while True:
            winner = self._get_winner(board)
            if winner != 0 or len([c for c in range(7) if board[0, c] == 0]) == 0:
                break
            move = self._heuristic_move(board, player)
            board = self._apply_move(board, move, player)
            player = -player

        final_winner = self._get_winner(board)
        if final_winner == 0:
            return 0.5
        return 1.0 if final_winner == state.player else 0.0

    def _heuristic_move(self, board, player):
        free_cols = [c for c in range(7) if board[0, c] == 0]
        # 1. Ganar inmediatamente
        for col in free_cols:
            row = self._get_drop_row(board, col)
            if row is not None and self._check_win_at(board, row, col, player):
                return col
        # 2. Bloquear victoria del oponente
        opponent = -player
        for col in free_cols:
            row = self._get_drop_row(board, col)
            if row is not None and self._check_win_at(board, row, col, opponent):
                return col
        # 3. Heurística de centro determinista (rápida)
        weights = [1, 2, 3, 4, 3, 2, 1]
        best_col = max(free_cols, key=lambda c: weights[c])
        return best_col

    # Versión optimizada de check_win_at (más rápida)
    def _check_win_at(self, board, r, c, player):
        rows, cols = board.shape
        # Horizontal
        count = 1
        for dc in (-1, 1):
            nc = c + dc
            while 0 <= nc < cols and board[r, nc] == player:
                count += 1
                nc += dc
        if count >= 4:
            return True
        # Vertical
        count = 1
        nr = r + 1
        while nr < rows and board[nr, c] == player:
            count += 1
            nr += 1
        if count >= 4:
            return True
        # Diagonal principal
        count = 1
        for dr, dc in ((-1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            while 0 <= nr < rows and 0 <= nc < cols and board[nr, nc] == player:
                count += 1
                nr += dr
                nc += dc
        if count >= 4:
            return True
        # Diagonal secundaria
        count = 1
        for dr, dc in ((-1, 1), (1, -1)):
            nr, nc = r + dr, c + dc
            while 0 <= nr < rows and 0 <= nc < cols and board[nr, nc] == player:
                count += 1
                nr += dr
                nc += dc
        if count >= 4:
            return True
        return False

    def _apply_move(self, board, col, player):
        new_board = board.copy()
        row = self._get_drop_row(new_board, col)
        if row is not None:
            new_board[row, col] = player
        return new_board

    def _get_drop_row(self, board, col):
        rows = board.shape[0]
        for r in range(rows-1, -1, -1):
            if board[r, col] == 0:
                return r
        return None

    # Versión optimizada de get_winner (sin all(), con bucles)
    def _get_winner(self, board):
        rows, cols = board.shape
        for r in range(rows):
            for c in range(cols):
                p = board[r, c]
                if p == 0:
                    continue
                # Horizontal
                if c + 3 < cols:
                    if board[r, c+1] == p and board[r, c+2] == p and board[r, c+3] == p:
                        return p
                # Vertical
                if r + 3 < rows:
                    if board[r+1, c] == p and board[r+2, c] == p and board[r+3, c] == p:
                        return p
                # Diagonal principal
                if r + 3 < rows and c + 3 < cols:
                    if board[r+1, c+1] == p and board[r+2, c+2] == p and board[r+3, c+3] == p:
                        return p
                # Diagonal secundaria
                if r + 3 < rows and c - 3 >= 0:
                    if board[r+1, c-1] == p and board[r+2, c-2] == p and board[r+3, c-3] == p:
                        return p
        return 0

    def _backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.wins += result
            result = 1.0 - result if result != 0.5 else 0.5
            node = node.parent

    def _result_from_perspective(self, final_state, perspective_player):
        winner = final_state.get_winner()
        if winner == 0:
            return 0.5
        return 1.0 if winner == perspective_player else 0.0


class Aha(Policy):
    def __init__(self, player: int = -1, num_simulations: int = 600):
        super().__init__()
        self.player = player
        self.num_simulations = num_simulations
        self.agent = None

    def mount(self, *args, **kwargs):
        self.agent = MCTSAgentHeuristic(
            player=self.player,
            num_simulations=self.num_simulations
        )
        if hasattr(self.agent, 'mount'):
            self.agent.mount(*args, **kwargs)

    def act(self, s: np.ndarray) -> int:
        if self.agent is None:
            self.mount()
        return self.agent.act(s)