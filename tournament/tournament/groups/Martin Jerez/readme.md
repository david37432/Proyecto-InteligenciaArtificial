# QLearningAgent — Martin Jerez

Agente Connect-4 basado en **Q-Learning offline + Alpha-Beta Minimax online**.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `policy.py` | Agente final. Pesos pre-entrenados embebidos, `mount()` instantáneo. |

No se necesitan archivos adicionales. Los pesos entrenados con 5 000 episodios
de auto-juego están embebidos directamente en `policy.py` como constante `_W`.

## Clase del agente

```
QLearningAgent  (hereda de Policy)
```

## Ejecución

```python
from policy import QLearningAgent

agent = QLearningAgent(player=-1)   # -1 = Rojo, 1 = Amarillo
agent.mount()                        # instantáneo (~0 ms)
col = agent.act(board)               # board: np.ndarray (6×7, dtype int)
```

`act()` devuelve un entero 0–6 (columna elegida).  
Tiempo de respuesta por movimiento: **< 50 ms** (Alpha-Beta depth=4).

## Dependencias

- `numpy` (única dependencia externa)
- `connect4.policy.Policy` (provista por el entorno del torneo)

## Parámetros opcionales

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `player` | `-1` | Color del agente |
| `search_depth` | `4` | Profundidad Alpha-Beta |
