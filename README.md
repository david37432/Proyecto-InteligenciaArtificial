# Agente RL para Connect-4 – Martin Jerez

**Curso:** Fundamentos de Inteligencia Artificial – Universidad de La Sabana, 2026.1

---

## Descripción

`QLearningAgent` es un agente de aprendizaje por refuerzo que juega Connect-4. A diferencia de los agentes MCTS del repositorio, **aprende offline** mediante auto-juego antes de la partida y actúa en **tiempo constante** durante el juego real (sin búsqueda en árbol).

---

## Fundamento teórico

### Competitive MDPs (Diapositiva 12)

Connect-4 se modela como un *Alternating Markov Game* zero-sum:

```
G = ⟨S, {A_i}², P, {r_i}², γ=1, η⟩
```

Recompensas solo al final: r ∈ {−1, 0, 1}.

### Truco bipolar

Para juegos zero-sum se puede aprender **una sola Q-function** válida para ambos jugadores. Al propagarse hacia atrás, la recompensa se niega en cada cambio de turno:

```
r ← −γ · r
```

### Cross-MDP Policy Iteration (self-play)

El agente itera mejorando su política contra sí mismo:

```
while not converged:
    jugar un episodio completo como ambos jugadores
    actualizar pesos con Q-learning + truco bipolar
```

### Q-function aproximada

Se usa una combinación lineal de features:

```
Q(s, a) ≈ w · φ(s, a)
```

con 8 features: victoria inmediata, bloqueo, control central, amenazas propias, bloqueo de amenazas, ventanas dobles, control fila inferior, bias.

---

## Uso

```python
from rl_agent import QLearningAgent

# Instanciar y entrenar
agent = QLearningAgent(player=-1, n_episodes=5000)
agent.mount()  # entrena aquí

# Actuar dado un tablero numpy (6×7)
col = agent.act(board)
```

Para persistir los pesos entrenados:

```python
agent = QLearningAgent(player=-1, n_episodes=5000, weights_file="weights.pkl")
agent.mount()  # entrena y guarda; la próxima vez carga directamente
```

---

## Parámetros configurables

| Parámetro | Valor por defecto | Descripción |
|-----------|------------------|-------------|
| `n_episodes` | 5000 | Episodios de auto-juego (variable principal de análisis) |
| `alpha` | 0.01 | Tasa de aprendizaje |
| `epsilon` | 0.2 | Exploración ε-greedy durante entrenamiento |
| `gamma` | 1.0 | Factor de descuento (γ=1 como en el curso) |
| `weights_file` | None | Ruta para guardar/cargar pesos entrenados |

---

## Resultados principales

| n_episodes | Win rate vs aleatorio |
|------------|----------------------|
| 100 | ~55% |
| 500 | ~65% |
| 1000 | ~70% |
| 2000 | ~75% |
| 5000 | ~80% |

Ver experimentos completos en `entrega.ipynb`.

---

## Diferencias frente a los agentes MCTS

| Aspecto | MCTSAgentRandom | QLearningAgent |
|---------|-----------------|----------------|
| Paradigma | Búsqueda online | Aprendizaje offline |
| Árbol de juego | Sí | No |
| Tiempo de acción | O(simulaciones) | O(1) |
| Variable de análisis | # simulaciones | # episodios |
| Fundamento | Diapo 13 | Diapo 12 + 13 |

---

## Archivos

```
├── rl_agent.py        # QLearningAgent (este agente)
├── mcts_random.py     # MCTSAgentRandom (agente base del master)
├── entrega.ipynb      # Experimentos y análisis
├── connect4/          # Framework del entorno
└── test_rl_agent.py   # Script de prueba rápida
```
