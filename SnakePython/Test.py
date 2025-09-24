"""
test_last_apple_time.py
Prueba rápida para verificar que el agente Snake no se quede atrapado con la última manzana,
y mide el tiempo de ejecución de cada partida.
"""

import time
from Snake_PyGame import SnakeEnv, choose_action_Astar

NUM_PARTIDAS = 10  # Número de partidas a probar

for i in range(NUM_PARTIDAS):
    env = SnakeEnv(size=10, apples_count=35)
    pasos = 0
    start_time = time.time()  # Inicio del conteo de tiempo

    while not env.done:
        siguiente = choose_action_Astar(env)
        if siguiente is None:
            break  # el agente no puede moverse, fin de juego
        env.step(siguiente)
        pasos += 1

    end_time = time.time()
    tiempo = round(end_time - start_time, 2)
    manzanas_recogidas = 35 - len(env.apples)
    exito = manzanas_recogidas == 35

    print(f"Partida {i+1}: Pasos = {pasos}, Manzanas = {manzanas_recogidas}, Tiempo = {tiempo}s, Éxito = {exito}")
