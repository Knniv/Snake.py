Este proyecto implementa un agente inteligente capaz de jugar Snake en un tablero de 10x10, recolectando hasta 35 manzanas sin colisionar.  
El agente utiliza un algoritmo de búsqueda informada (A*) combinado con una estrategia de seguridad basada en BFS para evitar quedar atrapado.  
Incluye además una visualización con Pygame para observar cómo se mueve la serpiente.

Algoritmos Implementados

A*(A-Star)
-Búsqueda informada usando la **distancia Manhattan** como heurística.
-Encuentra rutas cortas desde la cabeza de la serpiente hasta la manzana objetivo.

Estrategia de Seguridad (BFS)
-Antes de ir a por una manzana, el agente simula el movimiento para comprobar si después tendrá camino hacia la cola.
-Si no es seguro, sigue la cola de la serpiente usando BFS.

Beneficios
-Combina **eficiencia** (A*) con **seguridad** (BFS).
-Minimiza colisiones y maximiza las manzanas recolectadas.

El proyecto incluye pygame.py, que abre una ventana permitiendo observar paso a paso cómo el agente toma decisiones:
-Manzanas en rojo
-Cuerpo de la serpiente en verde
-Cabeza de la serpiente en celeste

Para poder ejecutarlo se necesitará tener instalado Python 3.8 o superior (recomendado 3.10+) y la dependencia de pygame (más de una forma en caso de que una no funcione):
abrir Windows PowerShell-> pip install pygame
			-> python -m pip install pygame
			-> py -m pip install pygame

Para ejecutarlo se debe descargar la carpate del repositorio GitHub, abrir el cmd o Windows PowerShell y escribir los siguentes comandos:
-cd ruta/donde/descargaste/SnakePython (copiar la ruta de la carpeta que está en la parte superior de la ventana cuando abre la carpeta)
Una vez en la carpeta:
 py Snake_PyGame.py
	o
 Python Snake_PyGame.py

Cada vez que el agente termina una partida, en la consola se muestra un resumen con:
-Juego terminado en [pasos] pasos, manzanas recogidas: [número], tiempo: [segundos]s

También está disponible Test.py para realizar 10 pruebas seguidas pero sin pygame, lo que hace que sea mucho más rápido