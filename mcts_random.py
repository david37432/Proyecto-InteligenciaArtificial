# groups/tu_nombre/mcts_random.py

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
        # No removemos nada aquí porque el movimiento ya fue extraído en _expand
        self.children.append(child)
        return child


class MCTSAgentRandom(Policy):
    """MCTS con simulaciones completamente aleatorias."""
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

        current = state
        while not current.is_final():
            move = random.choice(current.get_free_cols())
            current = current.transition(move)

        return self._result_from_perspective(current, node.parent.state.player)

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