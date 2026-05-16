# Proyecto Connect-4 - Laboratorio de Experimentación Empírica
**Asignatura:** Fundamentos de Inteligencia Artificial (2026-1)  
**Universidad:** Universidad de La Sabana  
**Estudiante:** Oscar David Vergara Moreno  
**Agente Evaluado:** `MCTSAgentOscarQ` (MCTS con Sesgo de Recompensa Q-Value)

---

## 1. Idea Principal del Agente y Diferenciación Técnica

Para el desarrollo de este reto, la arquitectura del agente se pensó desde el inicio para aportar una perspectiva metodológica e independiente a las soluciones convencionales, garantizando una propuesta autónoma y original en el equipo de trabajo:

* **Diferenciación Técnica:** A diferencia de los enfoques tradicionales que concentran el conocimiento experto en optimizar la fase final de *Simulación (Rollout)* mediante funciones de evaluación pesadas, **mi propuesta ataca directamente las fases tempranas de Selección y Expansión en la raíz del árbol**.
* **Aporte Algorítmico:** Modifiqué la ecuación convencional de UCB1 para inyectar un término de **Sesgo Q-Value** ($Q_{\text{prior}}$)

El término $\frac{Q(s, a)}{N_i}$ actúa como una guía táctica instantánea de conocimiento preexistente (*prior*). Cuando un nodo tiene muy pocas visitas ($N_i$ es bajo), la estimación posicional heurística predomina y obliga al algoritmo a explorar caminos lógicos, evitando desperdiciar presupuesto en ramas subóptimas. A medida que el nodo es explotado y acumula simulaciones reales ($N_i \to \infty$), el peso del sesgo decae asintóticamente y las estadísticas empíricas del MCTS toman el control absoluto de la decisión.

### Heurísticas Inyectadas en el Q-Value:
* **Victoria Inmediata (+10.0):** Identifica si existe un movimiento legal que cierra el juego en 4 en línea de forma inmediata (*1-ply lookahead*).
* **Bloqueo Crítico (+8.0):** Detecta si el oponente tiene un 3 en línea activo y prioriza colocar la ficha en la columna que frustra su victoria en el siguiente turno.
* **Control de Columna Central (+2.0):** Incentiva el posicionamiento en la columna 3, la cual es geométricamente óptima por participar en la mayor cantidad de combinaciones ganadoras posibles en el tablero de Connect-4.
* **Amenazas Abiertas (+2.0):** Recompensa la creación de configuraciones propias de 3 fichas que cuenten con espacios libres adyacentes para forzar respuestas defensivas del rival.

---

## 2. Variables de la Investigación Empírica

Siguiendo los lineamientos del laboratorio para analizar el agente como un sistema dinámico, el diseño experimental en el notebook se estructuró bajo las siguientes variables controladas:

* **Variable Numérica Independiente ($X_1$):** El presupuesto de cómputo del agente, medido mediante el número de simulaciones por turno: `num_simulations` $\in \{10, 50, 100\}$.
* **Variable de Control ($C_1$):** El switch booleano `use_q_bias`. Activado (`True`) corre mi propuesta con sesgo heurístico en selección, y desactivado (`False`) corre el algoritmo MCTS Puro (*Baseline* de control).
* **Variable Dependiente ($Y_1$):** La tasa de victorias (*Win Rate*) obtenida tras series cruzadas de partidas en el simulador oficial.
* **Oponente Base:** El agente aleatorio legal (`RandomPolicy`) provisto por el profesor para medir la velocidad de explotación del entorno.

---

## 3. Experimentos Realizados y Resultados Esperados

### Experimento 1: Tasa de Victorias vs Presupuesto de Cómputo
Se configuró un torneo balanceado donde el agente juega el 50% de las partidas como Rojo (inicia el juego) y el 50% como Amarillo (segundo jugador). 
* **Bajo Cómputo (10 sims):** El MCTS Puro no tiene suficientes muestras estadísticas para decidir bien y comete errores graves. Mi versión con Sesgo Q-Value compensa la falta de tiempo usando el conocimiento experto, ganando la gran mayoría de juegos desde el inicio.
* **Alto Cómputo (100 sims o más):** Ambas versiones convergen y destrozan al agente aleatorio con tasas cercanas al 100%. Esto demuestra que el MCTS Puro eventualmente alcanza al agente sesgado si se le da el presupuesto de tiempo necesario para hacer miles de rollouts.

### Experimento 2: Auto-juego y Ventaja Posicional (Asimetría)
Al enfrentar a `MCTSAgentOscarQ` contra sí mismo (Sesgo Q vs MCTS Puro) con un presupuesto fijo, los datos validan la teoría clásica de Connect-4: el jugador que tiene el turno inicial (Rojo, $-1$) cuenta con una ventaja geométrica tan alta que domina la distribución de victorias frente al jugador que responde (Amarillo, $1$).

---

## 4. Cuello de Botella Identificado y Propuesta de Mejora

**Problema Detectado:** Al inspeccionar los tiempos de ejecución del notebook, se observa que el agente reconstruye el árbol MCTS por completo desde cero en cada turno. Si una partida dura 30 movimientos, el algoritmo desperdicia miles de simulaciones tirando a la basura información de sub-árboles que ya exploró en jugadas anteriores.

**Propuesta de Mejora:** Implementar un mecanismo de **Reutilización de Árbol**. Al recibir el estado del tablero, el agente no debe instanciar un nodo raíz vacío, sino buscar si el estado actual coincide con alguno de los hijos directos del nodo del turno anterior. Al conservar ese sub-árbol, se preservan las visitas acumuladas, incrementando drásticamente la profundidad de búsqueda sin gastar más milisegundos de procesamiento.

---

## 5. Estructura del Directorio del Proyecto

Para evitar problemas de rutas relativas o fallos de compilación en Windows/macOS, el árbol de carpetas debe verse así:

```text
Proyecto-InteligenciaArtificial/
├── mcts_random.py                # Código base del repositorio
├── connect4/                     # Motor oficial provisto por el profesor
│   ├── policy.py                 # Clase abstracta base
│   └── connect_state.py          # Reglas del juego y cálculo de gravedad
└── Groups/
    └── Oscar David Vergara Moreno/
        ├── policy.py             # MI CÓDIGO (Contiene MCTSAgentOscarQ)
        ├── entrega.ipynb         # Mi notebook con los experimentos
        └── README.md             # Este documento
