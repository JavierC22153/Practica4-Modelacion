# Practica4

## Uso
```
pip install -r requirements.txt
streamlit run app.py              # aplicación configurable
python run_experiments.py --reps 30 --out resultados   # tablas CSV y gráficas PNG para el informe
```

## Archivos
| Archivo | Contenido |
|---|---|
| `matriz.csv` | Anexo A (40×40 minutos; vacío = sin conexión) |
| `network.py` | Lectura de la matriz, grafo dirigido, validación de rutas, incidentes (cambiar tiempo / cerrar) |
| `routing.py` | Dijkstra, A* con heurística ALT, Bellman-Ford, Floyd-Warshall |
| `simulation.py` | Simulación de eventos discretos (llegadas Poisson, cola FIFO, vehículos, ida, entrega, regreso) |
| `experiments.py` | Tablas de rutas/validación, comparación de algoritmos|
| `plots.py` | Grafo (sentidos y tiempos) y gráficas de resultados |
| `app.py` | Interfaz Streamlit |
| `run_experiments.py` | Genera todo lo necesario para el informe en `resultados/` |

## Decisiones de modelado
- **Grafo:** dirigido; arista u→v si la celda tiene un tiempo. Asimétrico (32 pares con un solo sentido o tiempos distintos).
- **Algoritmo elegido:** A* con heurística ALT (landmarks + desigualdad triangular), admisible y consistente → mismo costo óptimo que Dijkstra con menos nodos expandidos. Las otras tres técnicas están implementadas y se contrastan en pruebas y en la pestaña *Rutas*.
- **Ida y regreso** se calculan por separado (las calles de un solo sentido hacen que el inverso no siempre sea el mejor ni el válido).
- **Política "actual":** hojas de despacho + camino inverso. Si un tramo se cierra, el conductor se desvía por el camino más corto.
- **Política "algoritmo":** caminos mínimos recalculados con la red vigente (al cambiar la red se reconstruyen los landmarks).
- **Llegadas:** exponenciales con media 60/λ min hasta el cierre de recepción; destinos uniformes. Misma semilla ⇒ mismos pedidos para ambas políticas.
- **Utilización:** tiempo ocupado (ida + servicio + regreso) / (vehículos × tiempo hasta que el último vehículo regresa o se cierra la recepción, lo que ocurra después).
- **Tiempo adicional:** minutos después del cierre de recepción hasta la última entrega (en `regreso_adicional` queda hasta el último regreso).
- Se asume tiempo de entrega en el destino = 0 por defecto (configurable), tiempos de viaje deterministas y sin averías.
