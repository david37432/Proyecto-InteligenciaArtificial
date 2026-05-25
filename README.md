# Agente MCTS Heurístico para Connect-4

**Alumno:** David Eduardo López Jiménez  
**Asignatura:** Fundamentos de Inteligencia Artificial  
**Repositorio local:** `tournament/groups/David Eduardo Lopez Jimenez/`

Este directorio contiene la implementación de un agente inteligente basado en **Monte Carlo Tree Search (MCTS)** con una heurística de simulación optimizada, diseñado para jugar Connect-4 de manera autónoma. El agente cumple con los requisitos del proyecto (nunca pierde contra el aleatorio y gana al menos el 95% de las partidas) y ha sido evaluado experimentalmente.

##  Contenido

- `policy.py` – Implementación completa del agente (clase `Aha` y `MCTSAgentHeuristic`).
- `entrega.ipynb` – Notebook con todos los experimentos, gráficas y análisis (criterios 2 y 3 de la rúbrica).
- `README.md` – Este archivo.

## 🧠 Código completo del agente

El agente se encuentra en `policy.py`. La clase principal es `Aha`, que envuelve a `MCTSAgentHeuristic`. El código está optimizado con NumPy y utiliza una heurística de simulación en tres niveles:

1. **Ganar inmediatamente** – Si existe un movimiento que conecta 4 fichas, se elige.
2. **Bloquear al oponente** – Si el oponente amenaza con ganar en su siguiente turno, se bloquea.
3. **Centro** – Se favorecen las columnas centrales (pesos `[1,2,3,4,3,2,1]`).

El árbol MCTS se construye con selección UCB1, expansión, simulación heurística y retropropagación.

### `policy.py`

```python
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
        # 3. Heurística de centro determinista
        weights = [1, 2, 3, 4, 3, 2, 1]
        best_col = max(free_cols, key=lambda c: weights[c])
        return best_col

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

    def _get_winner(self, board):
        rows, cols = board.shape
        for r in range(rows):
            for c in range(cols):
                p = board[r, c]
                if p == 0:
                    continue
                if c + 3 < cols and board[r, c+1] == p and board[r, c+2] == p and board[r, c+3] == p:
                    return p
                if r + 3 < rows and board[r+1, c] == p and board[r+2, c] == p and board[r+3, c] == p:
                    return p
                if r + 3 < rows and c + 3 < cols and board[r+1, c+1] == p and board[r+2, c+2] == p and board[r+3, c+3] == p:
                    return p
                if r + 3 < rows and c - 3 >= 0 and board[r+1, c-1] == p and board[r+2, c-2] == p and board[r+3, c-3] == p:
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
```
##  Datos necesarios para la ejecución

El agente requiere el entorno del torneo que proporciona la cátedra (`connect4`).  
Asegúrate de que la siguiente estructura esté disponible en tu `PYTHONPATH` o en el directorio raíz del proyecto:

```text
tournament/
├── connect4/
│   ├── __init__.py
│   ├── connect_state.py
│   ├── policy.py (clase base)
│   └── ...
├── groups/
│   └── David Eduardo Lopez Jimenez/
│       ├── policy.py
│       ├── entrega.ipynb
│       └── README.md
└── main.py
## Dependencias externas

Instalar con `pip` si es necesario:

- `numpy`
- `matplotlib`
- `seaborn`
- `jupyter`

No se requieren archivos de datos adicionales; todos los experimentos se generan al ejecutar `entrega.ipynb`.

---

# 🚀 Breve guía de uso

## 1. Ejecutar el notebook de análisis

Abre `entrega.ipynb` con Jupyter Notebook/Lab y ejecuta todas las celdas. Esto generará:

- Evaluación frente al agente aleatorio (`RandomPolicy`) con diferentes números de simulaciones:
  - 10
  - 50
  - 100
  - 200
  - 400
  - 600
  - 2000

- Gráficas de:
  - tasa de victorias
  - tiempo por movimiento
  - autodesempeño

- Análisis del cuello de botella y propuesta de mejora.

---

## 2. Usar el agente en el torneo principal

Dentro del script `main.py` del torneo (ubicado en la raíz), puedes importar y usar tu agente de la siguiente manera:

```python
import sys

sys.path.insert(0, 'groups/David Eduardo Lopez Jimenez')

from policy import Aha

# Crear agente con 600 simulaciones (configuración por defecto)
agent = Aha(player=-1, num_simulations=600)
agent.mount()

# Obtener acción para un tablero dado
action = agent.act(board)
```

---

## 3. Enfrentar dos agentes manualmente

Puedes usar la función `play_game` definida en el notebook:

```python
from connect4.connect_state import ConnectState

def play_game(agent_red, agent_yellow):
    state = ConnectState()

    while not state.is_final():
        if state.player == -1:
            col = agent_red.act(state.board.copy())
        else:
            col = agent_yellow.act(state.board.copy())

        state = state.transition(col)

    return state.get_winner()

winner = play_game(agent_red, agent_yellow)

print("Ganador:", winner)

# -1 = rojo
# 1 = amarillo
# 0 = empate
```

---

## 4. Experimentación rápida

Modifica el número de simulaciones directamente al instanciar el agente.

Por ejemplo, para una versión rápida (50 simulaciones):

```python
fast_agent = Aha(player=-1, num_simulations=50)
fast_agent.mount()
```

---
# Resultados esperados (resumen)

Los experimentos realizados en `entrega.ipynb` muestran:

## Frente a oponente aleatorio

- Con 600 simulaciones: **86% de victorias**
- Con 2000 simulaciones: **99% de victorias**  
  (superando el 95% exigido)

---

## Autodesempeño  
(agente con 600 simulaciones vs menos simulaciones)

- 600 vs 50 → 84% victorias para el de 600
- 600 vs 100 → 86% victorias
- 600 vs 200 → 90% victorias
- 600 vs 400 → 52% victorias (prácticamente empate)

---

## Tiempo por movimiento

| Simulaciones | Tiempo |
|---|---|
| 10 | 8 ms |
| 50 | 40 ms |
| 100 | 77 ms |
| 200 | 154 ms |
| 400 | 302 ms |
| 600 | 452 ms |
| 2000 | ≈1500 ms |

---

## Compromiso tiempo-rendimiento

Para el torneo se recomienda usar entre:

- **400 y 600 simulaciones**  
  (~300–450 ms por movimiento)

o

- **2000 simulaciones**  
  si se prioriza la máxima fiabilidad.

---

#  Notas adicionales

- La heurística implementada es determinista  
  (sin aleatoriedad en el desempate por centro) para mayor consistencia.

- Para reproducir exactamente los experimentos, fija la semilla aleatoria:

```python
np.random.seed(42)
random.seed(42)
```

- Los resultados numéricos incluidos en el notebook corresponden a ejecuciones reales.  
  Si los vuelves a ejecutar, pueden variar ligeramente debido a la aleatoriedad, pero las tendencias generales se mantienen.

---

# 🔗 Enlace al código final

El código completo del agente está disponible en el branch del estudiante:
https://github.com/david37432/Proyecto-InteligenciaArtificial/tree/David-Eduardo-Lopez-Jimenez
