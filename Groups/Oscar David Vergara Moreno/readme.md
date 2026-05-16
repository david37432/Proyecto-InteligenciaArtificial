# Proyecto Connect-4 — Laboratorio de Experimentación Empírica

- **Asignatura:** Fundamentos de Inteligencia Artificial (2026-1)
- **Universidad:** Universidad de La Sabana
- **Estudiante:** Oscar David Vergara Moreno
- **Agente Evaluado:** `MCTSAgentOscarQ` (MCTS con Sesgo de Recompensa Q-Value)

---

## 🚀 Guía Rápida de Iniciación (Cómo probar el proyecto)

Si quieres ejecutar este laboratorio en tu computadora y ver las gráficas de rendimiento, sigue estos pasos desde la terminal:

### 1. Clonar el repositorio y entrar al proyecto

```bash
git clone <url-del-repositorio>
cd Proyecto-InteligenciaArtificial
```

### 2. Configurar el entorno virtual e instalar dependencias

Para asegurar que todo corra con las librerías aisladas sin romper nada en tu sistema (especialmente en Windows), ejecuta:

```bash
# Crear el entorno virtual
python -m venv .venv

# Activar el entorno virtual
# En Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# En Mac/Linux:
source .venv/bin/activate

# Instalar los paquetes necesarios para el Notebook
pip install ipykernel numpy matplotlib
```

### 3. Ejecutar los Experimentos

1. Abre VS Code en la raíz del proyecto: `code .`
2. Ve a la carpeta `Groups/Oscar David Vergara Moreno/` y abre el archivo `entrega.ipynb`.
3. Arriba a la derecha de VS Code, haz clic en **Select Kernel**, elige **Python Environments...** y selecciona el Python que está dentro de tu carpeta `.venv`.
4. Dale a **Run All** para simular las partidas y generar las gráficas automáticamente.

---

## 1. Idea Principal del Agente y Diferenciación Técnica

Para este laboratorio, diseñé la arquitectura de mi agente pensando en aportar un enfoque independiente y propio a las soluciones comunes, buscando que mi propuesta fuera original y autónoma en el equipo de trabajo:

- **Diferenciación Técnica:** En lugar de hacer lo tradicional (que es meterle lógica pesada a la fase final de **Simulación** o *Rollout* mediante funciones de evaluación complejas), mi propuesta ataca directamente las fases iniciales de **Selección** y **Expansión** desde la raíz del árbol.
- **Aporte Algorítmico:** Modifiqué la ecuación de selección estándar para meterle un término que llamé **"Sesgo Q-Value"**.

Este sesgo funciona como una guía táctica instantánea basada en el conocimiento del juego. Cuando un movimiento del árbol ha sido explorado muy pocas veces, la heurística toma el control y obliga al algoritmo a irse por caminos lógicos, evitando que gaste el tiempo en jugadas tontas. A medida que esa jugada se simula más y acumula datos reales, el peso de mi heurística va desapareciendo y las estadísticas puras del MCTS toman el control absoluto de la decisión.

### Heurísticas inyectadas en mi Sesgo Q

- **Victoria Inmediata (+10.0):** Escanea el tablero para ver si hay una jugada legal que complete las 4 fichas en línea en ese mismo instante y ganar de una.
- **Bloqueo Crítico (+8.0):** Revisa si el rival tiene 3 fichas alineadas y prioriza poner la ficha en la columna que le dañe el juego para el próximo turno.
- **Control de Columna Central (+2.0):** Le da puntos extra a la columna del centro (la 3), ya que geométricamente es la que conecta con la mayor cantidad de combinaciones ganadoras en Connect-4.
- **Amenazas Abiertas (+2.0):** Premia la creación de grupos de 3 fichas propias que tengan espacios libres a los lados para presionar al rival a defenderse.

---

## 2. Variables de la Investigación Empírica

Para analizar el agente como un sistema dinámico (tal como lo pide la guía del laboratorio), configuré las siguientes variables dentro del código del notebook:

- **Variable Independiente (X1):** El presupuesto de cómputo del agente, controlado a través del número de simulaciones por turno (`num_simulations`), probando con valores de **10, 30 y 70**.
- **Variable de Control (C1):** Un interruptor booleano llamado `use_q_bias`. Cuando está en `True` corre mi propuesta mejorada, y cuando está en `False` corre el MCTS Puro de toda la vida para tener un punto de comparación.
- **Variable Dependiente (Y1):** La tasa de victorias (*Win Rate*) que saca el agente después de jugar los torneos en el simulador.
- **Oponente Base:** El agente aleatorio (`RandomPolicy`) que nos dio el profesor para medir qué tan rápido explota el entorno nuestro algoritmo.

---

## 3. Experimentos Realizados y Resultados Obtenidos

### Experimento 1: Tasa de Victorias vs Presupuesto de Cómputo (Prueba contra Agente Aleatorio)

En este experimento medimos cómo le va a nuestro agente contra el oponente aleatorio usando un torneo balanceado de **16 partidas** (8 jugando de Rojo y 8 de Amarillo). Para que la gráfica mostrara el comportamiento real del MCTS sin "ayudas" externas, desactivamos temporalmente las funciones de bloqueo y victoria inmediata de un solo paso:

- **Bajo Cómputo (10 simulaciones):** El MCTS Puro (Baseline) se queda corto y saca una tasa de victorias del **87.5%**. Esto pasa porque con tan poquitas simulaciones el árbol comete errores ciegos y el aleatorio le logra ganar un par de partidas. En cambio, mi versión **MCTS + Sesgo Q-Value mantiene un 100.0% perfecto**. Esto demuestra que la heurística en la fórmula de selección guía al agente a jugar de forma lógica desde el principio, compensando la falta de tiempo.
- **Alto Cómputo (30 y 70 simulaciones):** Aquí ambas versiones alcanzan el **100.0%** de victorias. El resultado es lógico porque, si le damos más presupuesto de tiempo al MCTS tradicional, la fuerza bruta estadística del árbol puro termina encontrando las jugadas correctas e iguala la ventaja táctica que mi agente tenía desde el inicio.

### Experimento 2: Auto-Juego (Enfrentamiento Directo)

Para evaluar qué tanto aporta mi diseño cuando los recursos son muy limitados, pusimos a pelear a `MCTSAgentOscarQ` contra el MCTS Puro (Baseline). Configuré un presupuesto fijo de apenas **50 simulaciones por turno** para cada uno y los pusimos a jugar cara a cara.

Los resultados de las partidas fueron contundentes:

- **Victorias MCTS + Sesgo Q-Value:** **83.3%** (5 partidas ganadas).
- **Victorias MCTS Puro (Baseline):** 16.7% (1 partida ganada).
- **Empates:** 0.0%.

**Análisis para la Sustentación:** Este resultado es clave porque demuestra que mi agente aprovecha mucho mejor el procesador. Con solo 50 simulaciones por turno, el MCTS Puro no alcanza a llenar el árbol y toma decisiones casi a la suerte en las primeras jugadas. Mi algoritmo usa el sesgo como un filtro: cuando un nodo casi no se ha visitado, la heurística obliga al árbol a concentrar esas 50 simulaciones solo en las columnas buenas (como el centro o los bloqueos) en vez de perder tiempo explorando jugadas absurdas. Por eso termina acorralando al modelo base en la mayoría de partidas.

---

## 4. Cuello de Botella Identificado y Propuesta de Mejora

**Problema Detectado:** Al revisar cómo corre el notebook, me di cuenta de que el agente borra y vuelve a construir el árbol MCTS desde cero en cada turno. Si una partida dura 30 jugadas, el algoritmo desperdicia un montón de información valiosa tirando a la basura sub-árboles enteros que ya había explorado y calculado en los turnos pasados.

**Propuesta de Mejora:** La solución ideal sería implementar un sistema de **Reutilización de Árbol**. Cuando sea nuestro turno de jugar, en vez de crear una raíz vacía, el agente debería buscar en la memoria si el estado actual del tablero coincide con alguno de los nodos hijos que calculó en el turno anterior. Al reusar esa parte del árbol, mantendríamos las visitas y los datos acumulados, logrando que el agente busque mucho más profundo sin gastar más milisegundos de procesamiento.

---

## 5. Estructura del Directorio del Proyecto

Para que el proyecto compile sin problemas de rutas en Windows o Mac, la organización de las carpetas quedó de la siguiente manera:

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
```
