"""
snake_agent_pygame.py
Versión con visualización usando pygame del agente Snake.
- Tablero 10x10.
- Hasta 35 manzanas.
- Agente usa A* y estrategia de seguridad.
- Incluye comentarios detallados en español neutro.

Requisitos: pip install pygame
Ejecución: python3 snake_agent_pygame.py
"""

import random
import time
from collections import deque
import heapq
import pygame

# ---------------- Parámetros del juego ----------------
BOARD_SIZE = 10       # tamaño del tablero (10x10)
CELL_SIZE = 40        # tamaño de cada celda en pixeles (para pygame)
MAX_APPLES = 35       # número máximo de manzanas
MIN_START_LENGTH = 1  # longitud mínima inicial de la serpiente
FPS = 30              # velocidad de refresco del juego en frames por segundo

# direcciones posibles (arriba, abajo, izquierda, derecha)
DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

# --------- Funciones auxiliares ---------
def manhattan(a,b):
    """Distancia de Manhattan entre dos celdas."""
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# ---------------- Clase del entorno (Snake) ----------------
class SnakeEnv:
    """
    Clase que representa el entorno del juego Snake.
    Se encarga de:
    - Almacenar la posición de la serpiente y manzanas
    - Aplicar las reglas del juego (mover, comer, colisiones)
    - Verificar fin de partida
    """
    def __init__(self, size=BOARD_SIZE, apples_count=None, seed=None):
        if seed is not None:
            random.seed(seed)
        self.size = size
        self.apples_count = apples_count if apples_count is not None else random.randint(1, MAX_APPLES)
        self.reset()

    def reset(self):
        """Inicializa el tablero con la serpiente y manzanas."""
        mid = self.size // 2
        # Serpiente como deque (cola doble) para añadir/quitar cabeza y cola eficientemente
        self.snake = deque()
        for i in range(MIN_START_LENGTH):
            # La cabeza queda al frente
            self.snake.appendleft((mid, mid - i))
        self.direction = (0,1)
        # Generamos manzanas en posiciones libres
        self.apples = set()
        free = [(r,c) for r in range(self.size) for c in range(self.size) if (r,c) not in self.snake]
        random.shuffle(free)
        for pos in free[:min(self.apples_count, len(free))]:
            self.apples.add(pos)
        self.steps = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        """Retorna estado actual (para depuración)."""
        return {
            'snake': list(self.snake),
            'apples': set(self.apples),
            'size': self.size
        }

    def in_bounds(self, pos):
        """Chequea si la posición está dentro del tablero."""
        r,c = pos
        return 0 <= r < self.size and 0 <= c < self.size

    def is_collision(self, pos, ignore_tail=False):
        """Verifica si pos colisiona con paredes o cuerpo."""
        if not self.in_bounds(pos):
            return True
        body = set(self.snake)
        if ignore_tail:
            tail = self.snake[-1]
            body = body - {tail}
        return pos in body

    def step(self, next_pos):
        """
        Mueve la serpiente hacia next_pos.
        Controla:
        - Comer manzana
        - Crecer o mover cola
        - Fin de juego por colisión
        """
        if not self.in_bounds(next_pos):
            self.done = True
            return -1, True
        if next_pos in list(self.snake)[:-1]:  # colisión con cuerpo (excepto cola)
            self.done = True
            return -1, True
        # mover cabeza
        self.snake.appendleft(next_pos)
        ate = next_pos in self.apples
        if ate:
            self.apples.remove(next_pos)
            # no movemos cola => crece
        else:
            self.snake.pop()  # movemos cola
        self.steps += 1
        if len(self.apples) == 0:
            self.done = True
            return 1, True
        return 0, False

    def head(self):
        return self.snake[0]

    def tail(self):
        return self.snake[-1]

# ---------------- A* Search ----------------
def astar_search(start, goal, snake_body, board_size, allow_tail=True):
    """
    Algoritmo A* en la grilla desde start hasta goal evitando snake_body.
    allow_tail=True: trata la cola como libre (porque se mueve).
    Retorna lista de posiciones desde start (excluido) hasta goal (incluido).
    """
    blocked = set(snake_body)
    if allow_tail and len(snake_body) > 0:
        tail = snake_body[-1]
        blocked = set(snake_body) - {tail}

    def neighbors(pos):
        for d in DIRS:
            np = (pos[0]+d[0], pos[1]+d[1])
            if 0 <= np[0] < board_size and 0 <= np[1] < board_size:
                if np not in blocked:
                    yield np

    open_heap = []
    heapq.heappush(open_heap, (manhattan(start,goal), 0, start))
    came_from = {}
    gscore = {start:0}

    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        if current == goal:
            # reconstruir camino
            path = []
            node = current
            while node != start:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path
        for nb in neighbors(current):
            tentative_g = g + 1
            if nb not in gscore or tentative_g < gscore[nb]:
                gscore[nb] = tentative_g
                came_from[nb] = current
                fscore = tentative_g + manhattan(nb, goal)
                heapq.heappush(open_heap, (fscore, tentative_g, nb))
    return None

# ---------------- BFS para fallback ----------------
from collections import deque as dq
def bfs_path(start, goal, snake_body, board_size, allow_tail=True):
    """Busca camino más corto con BFS (usado para seguir cola si no hay apple segura)."""
    blocked = set(snake_body)
    if allow_tail and len(snake_body) > 0:
        tail = snake_body[-1]
        blocked = set(snake_body) - {tail}
    queue = dq([start])
    prev = {start: None}
    while queue:
        cur = queue.popleft()
        if cur == goal:
            path = []
            node = cur
            while node != start:
                path.append(node)
                node = prev[node]
            path.reverse()
            return path
        for d in DIRS:
            nb = (cur[0]+d[0], cur[1]+d[1])
            if 0 <= nb[0] < board_size and 0 <= nb[1] < board_size and nb not in blocked and nb not in prev:
                prev[nb] = cur
                queue.append(nb)
    return None

# ---------------- Política del agente ----------------
def choose_action_Astar(env):
    """
    Selecciona la siguiente celda para la cabeza usando A* hacia la manzana más cercana.
    Incluye alternativa simple para la última manzana, evitando bucles.
    """
    head = env.head()
    body = list(env.snake)
    apples = list(env.apples)

    if not apples:
        return None

    # Ordena manzanas por distancia Manhattan
    apples.sort(key=lambda a: manhattan(head, a))

    for apple in apples:
        path = astar_search(head, apple, body, env.size, allow_tail=True)
        if path:
            # Simular si al final queda camino hasta la cola
            sim_body = deque(body)
            for step_pos in path:
                sim_body.appendleft(step_pos)
                if step_pos == apple:
                    pass  # crece
                else:
                    sim_body.pop()
            new_head = sim_body[0]
            new_tail = sim_body[-1]
            p = bfs_path(new_head, new_tail, list(sim_body), env.size, allow_tail=True)
            if p is not None:
                return path[0]

    # Fallback general: último recurso, cualquier movimiento válido
    for d in DIRS:
        cand = (head[0]+d[0], head[1]+d[1])
        if not env.is_collision(cand, ignore_tail=True):
            return cand

    # Si no hay movimientos posibles
    return None

# ---------------- Función principal con pygame ----------------
def run_game():
    pygame.init()
    width = BOARD_SIZE * CELL_SIZE
    height = BOARD_SIZE * CELL_SIZE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Snake AI A*")
    clock = pygame.time.Clock()

    env = SnakeEnv(size=BOARD_SIZE, apples_count=MAX_APPLES)
    start_time = time.perf_counter()

    running = True
    while running and not env.done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        next_pos = choose_action_Astar(env)
        if next_pos is None:
            break
        env.step(next_pos)

        # -------- Dibujo con pygame --------
        screen.fill((0,0,0))  # fondo negro

        # manzanas en rojo
        for a in env.apples:
            pygame.draw.rect(screen, (255,0,0), (a[1]*CELL_SIZE, a[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # cuerpo serpiente en verde
        for i,seg in enumerate(env.snake):
            color = (0,255,0) if i>0 else (0,200,255)  # cabeza en celeste
            pygame.draw.rect(screen, color, (seg[1]*CELL_SIZE, seg[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        pygame.display.flip()
        clock.tick(FPS)  # controla la velocidad

    elapsed = time.perf_counter() - start_time
    print(f"Juego terminado en {env.steps} pasos, manzanas recogidas: {MAX_APPLES - len(env.apples)}, tiempo: {elapsed:.2f}s")
    pygame.quit()

if __name__ == "__main__":
    run_game()