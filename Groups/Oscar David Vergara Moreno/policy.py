
import math
import random
import time
import numpy as np

from connect4.connect_state import ConnectState
from connect4.policy import Policy

ROWS: int = 6
COLS: int = 7
CENTER_COL: int = 3

Q_WIN: float = 10.0    
Q_BLOCK: float = 8.0   
Q_BONUS: float = 2.0   

class MCTSNode:
    def __init__(self, state: ConnectState, parent=None, move=None,
                 q_prior: float = 0.0):
        self.state: ConnectState = state
        self.parent = parent
        self.move = move
        self.children: list = []
        self.visits: int = 0
        self.wins: float = 0.0
        self.untried_moves: list = state.get_free_cols()

        self.q_prior: float = q_prior

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.state.is_final()

    def add_child(self, move: int, child_state: ConnectState,
                  q_prior: float = 0.0) -> "MCTSNode":
        child = MCTSNode(child_state, parent=self, move=move, q_prior=q_prior)
        self.children.append(child)
        return child
    
class MCTSAgentOscarQ(Policy):

    def __init__(self, num_simulations: int = 600,
                 c_param: float = math.sqrt(2),
                 use_q_bias: bool = True,
                 seed=None):
        self.num_simulations: int = int(num_simulations)
        self.c_param: float = float(c_param)
        self.use_q_bias: bool = bool(use_q_bias)
        self.action_timeout = None
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._py_rng: random.Random = random.Random(
            int(self._rng.integers(0, 2**31 - 1))
        )
    def mount(self, action_timeout=None) -> None:
        self.action_timeout = action_timeout
        return None
    def act(self, s: np.ndarray) -> int:
        current_player: int = self._infer_player_to_move(s)
        winning_move = self._find_immediate_winning_move(s, current_player)
        if winning_move is not None:
            return int(winning_move)
        blocking_move = self._find_immediate_winning_move(s, -current_player)
        if blocking_move is not None:
            return int(blocking_move)

        root_state = ConnectState(board=s, player=current_player)
        root_node = MCTSNode(state=root_state, parent=None,move=None, q_prior=0.0)
        deadline = None
        if self.action_timeout is not None and self.action_timeout > 0:
            deadline = time.monotonic() + 0.85 * float(self.action_timeout)

        for i in range(self.num_simulations):
            self._run_one_iteration(root_node)
            if deadline is not None and (i & 15) == 15:
                if time.monotonic() >= deadline:
                    break
        if not root_node.children:
            free_cols = root_state.get_free_cols()
            return int(self._py_rng.choice(free_cols))

        best_child = max(root_node.children, key=lambda c: c.visits)
        return int(best_child.move)

    def _run_one_iteration(self, root: MCTSNode) -> None:

        node: MCTSNode = root
        while (not node.is_terminal()) and node.is_fully_expanded() and node.children:
            node = self._select_best_child(node)
        if (not node.is_terminal()) and (not node.is_fully_expanded()):
            node = self._expand(node)
        winner: int = self._simulate_rollout(node.state)
        self._backpropagate(node, winner)
    def _select_best_child(self, node: MCTSNode) -> MCTSNode:
        log_parent: float = math.log(node.visits) if node.visits > 0 else 0.0

        best_score: float = -float("inf")
        best_child: MCTSNode = node.children[0]

        for child in node.children:
            if child.visits == 0:
                return child

            exploitation: float = child.wins / child.visits
            exploration: float = self.c_param * math.sqrt(
                log_parent / child.visits
            )

            if self.use_q_bias:

                q_bonus: float = child.q_prior / child.visits
            else:

                q_bonus = 0.0

            ucb_score: float = exploitation + exploration + q_bonus
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child

        return best_child


    def _expand(self, node: MCTSNode) -> MCTSNode:
        idx: int = self._py_rng.randrange(len(node.untried_moves))
        move: int = int(node.untried_moves.pop(idx))
        mover: int = int(node.state.player)
        q_prior: float = self._evaluate_q_value(node.state.board, move, mover)
        new_state: ConnectState = node.state.transition(move)
        return node.add_child(move=move, child_state=new_state, q_prior=q_prior)
    def _simulate_rollout(self, state: ConnectState) -> int:
        board: np.ndarray = state.board.copy()
        player: int = int(state.player)
        winner_now: int = self._fast_winner_check(board)
        if winner_now != 0:
            return winner_now
        if self._is_full(board):
            return 0

        while True:
            free_cols = [c for c in range(COLS) if board[0, c] == 0]
            if not free_cols:
                return 0 

            col: int = free_cols[self._py_rng.randrange(len(free_cols))]

            last_r: int = -1
            for r in range(ROWS - 1, -1, -1):
                if board[r, col] == 0:
                    board[r, col] = player
                    last_r = r
                    break
            if self._is_winning_move(board, last_r, col, player):
                return player
            player = -player

    def _backpropagate(self, node: MCTSNode, winner: int) -> None:
        current = node
        while current is not None:
            current.visits += 1
            if current.parent is not None:
                mover_at_this_node: int = int(current.parent.state.player)
                if winner == mover_at_this_node:
                    current.wins += 1.0
                elif winner == 0:
                    current.wins += 0.5
            current = current.parent


    def _evaluate_q_value(self, board: np.ndarray, move: int, player: int) -> float:
        row: int = self._next_open_row(board, move)
        if row < 0:
            return 0.0
        board[row, move] = player
        is_winning_now: bool = self._is_winning_move(board, row, move, player)
        board[row, move] = 0  # restaurar siempre
        if is_winning_now:
            return Q_WIN
        opponent: int = -player
        board[row, move] = opponent
        opp_would_win: bool = self._is_winning_move(board, row, move, opponent)
        board[row, move] = 0
        if opp_would_win:
            return Q_BLOCK

        bonus: float = 0.0
        if move == CENTER_COL:
            bonus += Q_BONUS
        board[row, move] = player
        threats: int = self._count_open_three_threats(board, player)
        board[row, move] = 0
        if threats >= 1:
            bonus += Q_BONUS
        return min(bonus, 2.0 * Q_BONUS)

    def _next_open_row(self, board: np.ndarray, col: int) -> int:
        """Devuelve la fila donde caería una ficha en `col` (o -1 si llena)."""
        for r in range(ROWS - 1, -1, -1):
            if board[r, col] == 0:
                return r
        return -1

    def _is_full(self, board: np.ndarray) -> bool:
        """¿El tablero está completamente lleno? (fila superior sin ceros)."""
        return bool(np.all(board[0, :] != 0))

    def _is_winning_move(self, board: np.ndarray, r: int, c: int, player: int) -> bool:
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))
        for dr, dc in directions:
            count: int = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr, cc] == player:
                count += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr, cc] == player:
                count += 1
                rr -= dr
                cc -= dc
            if count >= 4:
                return True
        return False

    def _fast_winner_check(self, board: np.ndarray) -> int:

        for r in range(ROWS):
            for c in range(COLS - 3):
                s: int = int(board[r, c] + board[r, c+1]
                             + board[r, c+2] + board[r, c+3])
                if s == 4:
                    return 1
                if s == -4:
                    return -1
        for c in range(COLS):
            for r in range(ROWS - 3):
                s = int(board[r, c] + board[r+1, c]
                        + board[r+2, c] + board[r+3, c])
                if s == 4:
                    return 1
                if s == -4:
                    return -1
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                s = int(board[r, c] + board[r+1, c+1]
                        + board[r+2, c+2] + board[r+3, c+3])
                if s == 4:
                    return 1
                if s == -4:
                    return -1
        for r in range(ROWS - 3):
            for c in range(3, COLS):
                s = int(board[r, c] + board[r+1, c-1]
                        + board[r+2, c-2] + board[r+3, c-3])
                if s == 4:
                    return 1
                if s == -4:
                    return -1
        return 0

    def _count_open_three_threats(self, board: np.ndarray,
                                  player: int) -> int:
        threats: int = 0

    
        for r in range(ROWS):
            for c in range(COLS - 3):
                p_cnt = 0
                z_cnt = 0
                for k in range(4):
                    v = int(board[r, c + k])
                    if v == player:
                        p_cnt += 1
                    elif v == 0:
                        z_cnt += 1
                if p_cnt == 3 and z_cnt == 1:
                    threats += 1

        for c in range(COLS):
            for r in range(ROWS - 3):
                p_cnt = 0
                z_cnt = 0
                for k in range(4):
                    v = int(board[r + k, c])
                    if v == player:
                        p_cnt += 1
                    elif v == 0:
                        z_cnt += 1
                if p_cnt == 3 and z_cnt == 1:
                    threats += 1

      
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                p_cnt = 0
                z_cnt = 0
                for k in range(4):
                    v = int(board[r + k, c + k])
                    if v == player:
                        p_cnt += 1
                    elif v == 0:
                        z_cnt += 1
                if p_cnt == 3 and z_cnt == 1:
                    threats += 1


        for r in range(ROWS - 3):
            for c in range(3, COLS):
                p_cnt = 0
                z_cnt = 0
                for k in range(4):
                    v = int(board[r + k, c - k])
                    if v == player:
                        p_cnt += 1
                    elif v == 0:
                        z_cnt += 1
                if p_cnt == 3 and z_cnt == 1:
                    threats += 1

        return threats

    def _find_immediate_winning_move(self, board: np.ndarray,player: int):

        for c in range(COLS):
            if board[0, c] != 0:
                continue
            r: int = self._next_open_row(board, c)
            if r < 0:
                continue
            board[r, c] = player
            wins: bool = self._is_winning_move(board, r, c, player)
            board[r, c] = 0
            if wins:
                return c
        return None

    def _infer_player_to_move(self, s: np.ndarray) -> int:
        red_count: int = int(np.sum(s == -1))
        yellow_count: int = int(np.sum(s == 1))
        return -1 if red_count == yellow_count else 1
