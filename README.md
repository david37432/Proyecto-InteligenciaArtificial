# Agente Connect-4: Q-Learning + Alpha-Beta Minimax — Martin Jerez

**Curso:** Fundamentos de Inteligencia Artificial · Universidad de La Sabana · 2026.1  
**Branch:** [`Martin-Jerez`](https://github.com/david37432/Proyecto-InteligenciaArtificial/tree/Martin-Jerez)

---

## Idea principal

El `QLearningAgent` combina **aprendizaje offline** y **búsqueda online** en dos fases:

1. **Fase offline — entrenamiento (una sola vez antes de la partida):**  
   Auto-juego con el *truco bipolar* (Competitive MDPs, Diap. 12) aprende los pesos
   `w ∈ ℝ⁸` de la función de valor `V(s) = w · ψ(s)`.

2. **Fase online — decisión por movimiento:**  
   Alpha-Beta Minimax profundidad 4 usa `V(s)` como heurística de hoja.
   El árbol detecta forks y dobles amenazas que un lookup plano o un rollout
   aleatorio no pueden ver.

**Diferencia clave frente a los otros agentes del grupo:**

| Aspecto | Oscar (MCTS + Q-bias) | David (MCTS heurístico) | **Este agente** |
|---------|----------------------|------------------------|----------------|
| Aprendizaje | Ninguno | Ninguno | **Q-Learning offline** |
| Búsqueda online | MCTS rollout | MCTS rollout heurístico | **Alpha-Beta depth=4** |
| Heurística de evaluación | Fija (Q_WIN/Q_BLOCK) | Fija (ganar/bloquear/centro) | **Aprendida** (w·ψ(s)) |
| Fundamento del curso | Diap. 13 | Diap. 13 | **Diap. 12 + 13** |

---

## Versiones del agente

### v2 — Entrenamiento dinámico (versión recomendada)

**Código:** [`rl_agent.py`](https://github.com/david37432/Proyecto-InteligenciaArtificial/blob/Martin-Jerez/rl_agent.py)  
**Entrega torneo:** [`tournament/tournament/groups/Martin Jerez/policy.py`](https://github.com/david37432/Proyecto-InteligenciaArtificial/blob/Martin-Jerez/tournament/tournament/groups/Martin%20Jerez/policy.py)

Entrena los pesos al llamar `mount()`. Si se provee `weights_file`, los guarda y
reutiliza en ejecuciones posteriores.

```python
from rl_agent import QLearningAgent

# Crear y entrenar (corre ~30 s con 5000 episodios)
agent = QLearningAgent(player=-1, n_episodes=5000, search_depth=4)
agent.mount()

# Actuar dado un tablero numpy (6×7, dtype int)
col = agent.act(board)
```

Con caché de pesos (evita reentrenar):

```python
agent = QLearningAgent(
    player=-1,
    n_episodes=5000,
    search_depth=4,
    weights_file="weights.pkl"   # guarda/carga automáticamente
)
agent.mount()
col = agent.act(board)
```

### v2-hardcoded — Pesos pre-entrenados fijos

**Código:** [`tournament/tournament/groups/Martin Jerez/policy_hardcoded.py`](https://github.com/david37432/Proyecto-InteligenciaArtificial/blob/Martin-Jerez/tournament/tournament/groups/Martin%20Jerez/policy_hardcoded.py)

Contiene los pesos entrenados con 5 000 episodios grabados como constante.
`mount()` es instantáneo — no hay entrenamiento en runtime.
Útil para entornos donde el tiempo de inicialización es limitado.

```python
from policy_hardcoded import QLearningAgent   # pesos ya incluidos

agent = QLearningAgent(player=-1)
agent.mount()   # instantáneo (~0 ms)
col = agent.act(board)
```

---

## Parámetros configurables (v2 dinámica)

| Parámetro | Valor por defecto | Descripción |
|-----------|------------------|-------------|
| `player` | `-1` | Color del agente (−1 = Rojo, 1 = Amarillo) |
| `n_episodes` | `5000` | Episodios de auto-juego para el entrenamiento offline |
| `search_depth` | `4` | Profundidad del árbol Alpha-Beta durante el juego |
| `alpha` | `0.01` | Tasa de aprendizaje Q-Learning |
| `epsilon` | `0.3` | ε inicial (decae linealmente hasta `epsilon_end`) |
| `epsilon_end` | `0.05` | ε final al término del entrenamiento |
| `gamma` | `1.0` | Factor de descuento (γ=1 como en el curso) |
| `weights_file` | `None` | Ruta `.pkl` para guardar/cargar pesos entrenados |

---

## Fundamento teórico

### Competitive MDPs — truco bipolar (Diap. 12)

Connect-4 es un *Alternating Markov Game* zero-sum. Para juegos de este tipo se puede
aprender una sola función de valor `V` válida para ambos jugadores propagando hacia atrás
con el **truco bipolar**: al cambiar de turno, la recompensa se niega:

```
r ← −γ · r
```

La actualización de pesos en cada paso:

```
Δw = α · (r − w · ψ(s)) · ψ(s)
```

### Features ψ(s) — 8 dimensiones

| Índice | Feature |
|--------|---------|
| ψ₀ | Amenazas propias (ventanas 3 fichas + 1 libre) |
| ψ₁ | Amenazas del oponente |
| ψ₂ | Dobles propias (ventanas 2 fichas + 2 libres) |
| ψ₃ | Dobles del oponente |
| ψ₄ | Control columnas centrales (propio, ponderado) |
| ψ₅ | Control columnas centrales (oponente) |
| ψ₆ | Piezas en fila inferior (estabilidad) |
| ψ₇ | Bias constante |

### Alpha-Beta Minimax online (Diap. 13)

Orden de movimientos para maximizar la poda: movimientos ganadores → bloqueos → centro.
Detección de victoria forzada (score > 9 000) permite corte temprano.

---

## Resultados principales (ver análisis completo en `entrega.ipynb`)

| Configuración | vs Aleatorio | vs MCTS-200 | vs sí mismo (fuerte vs débil) |
|--------------|-------------|------------|-------------------------------|
| 5000 ep, depth=4 (final) | **>95%** | ~65% | — |
| 500 ep, depth=4 | ~80% | ~35% | 25% (frente al fuerte) |
| 5000 ep, depth=0 (sin búsqueda) | ~80% | ~30% | — |

---

## Ejecución rápida

```bash
# Desde la raíz del branch
python test_rl_agent.py          # enfrenta el agente vs Random (N partidas)
python get_weights.py            # pre-entrena y guarda pesos en weights.pkl
```

---

## Estructura de archivos

```
Martin-Jerez/
├── rl_agent.py                          # Agente v2 (entrenamiento dinámico)
├── mcts_random.py                       # Baseline MCTS aleatorio (comparación)
├── entrega.ipynb                        # Experimentos y gráficas del análisis
├── get_weights.py                       # Script para pre-entrenar y guardar pesos
├── test_rl_agent.py                     # Prueba rápida vs agente aleatorio
├── connect4/                            # Framework del entorno (no modificado)
└── tournament/tournament/groups/
    └── Martin Jerez/
        ├── policy.py                    # Entrega torneo (v2 dinámica)
        └── policy_hardcoded.py          # Entrega torneo (v2 pesos fijos)
```
