from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from network import DESTINOS

PUNTUAL_MIN = 75.0


@dataclass(frozen=True)
class Order:
    id: int
    arrival: float   # minutos desde el inicio de la jornada
    dest: str


def generate_orders(rate_per_hour, duration_hours, seed, destinos=DESTINOS):
    if rate_per_hour <= 0 or duration_hours <= 0:
        return []
    ss = np.random.SeedSequence(int(seed))
    rng_t, rng_d = (np.random.default_rng(s) for s in ss.spawn(2))
    mean_gap = 60.0 / rate_per_hour
    horizon = duration_hours * 60.0
    orders, t = [], 0.0
    while True:
        t += rng_t.exponential(mean_gap)
        if t > horizon:
            break
        orders.append(Order(len(orders) + 1, t, destinos[int(rng_d.integers(len(destinos)))]))
    return orders


@dataclass
class SimResult:
    records: list = field(default_factory=list)       # un dict por pedido
    queue_series: list = field(default_factory=list)  # [(t, largo de cola)]
    metrics: dict = field(default_factory=dict)
    n_vehicles: int = 0
    duration_min: float = 0.0


def simulate(orders, plans, n_vehicles, duration_min, service_min=0.0, limit_min=PUNTUAL_MIN):
    if n_vehicles < 1:
        raise ValueError("Se necesita al menos un vehículo")

    RETURN, DELIVERY, ARRIVAL = 0, 1, 2
    events, seq = [], 0

    def push(t, prio, kind, payload):
        nonlocal seq
        heapq.heappush(events, (t, prio, seq, kind, payload))
        seq += 1

    for o in orders:
        push(o.arrival, ARRIVAL, "arrival", o)

    free = list(range(1, n_vehicles + 1))
    heapq.heapify(free)
    queue = deque()
    records = {}
    qseries = [(0.0, 0)]
    busy = 0.0
    last_delivery = 0.0
    last_return = 0.0

    while events:
        t, _, _, kind, payload = heapq.heappop(events)

        if kind == "arrival":
            o = payload
            plan = plans.get(o.dest)
            if plan is None or not plan.reachable:
                records[o.id] = dict(id=o.id, destino=o.dest, llegada=o.arrival, estado="sin ruta")
            else:
                queue.append(o)

        elif kind == "delivery":
            o, v = payload
            last_delivery = max(last_delivery, t)
            push(t + plans[o.dest].back_min, RETURN, "return", v)

        else:  # return
            heapq.heappush(free, payload)
            last_return = max(last_return, t)

        while queue and free:
            o = queue.popleft()
            v = heapq.heappop(free)
            plan = plans[o.dest]
            delivered = t + plan.out_min + service_min
            response = delivered - o.arrival
            records[o.id] = dict(
                id=o.id, destino=o.dest, llegada=o.arrival, inicio_viaje=t,
                espera=t - o.arrival, ida=plan.out_min, entrega=delivered,
                solicitud_a_entrega=response, regreso=plan.back_min,
                vehiculo_libre=delivered + plan.back_min, vehiculo=v,
                puntual=response <= limit_min + 1e-9, estado="entregado",
            )
            busy += plan.out_min + service_min + plan.back_min
            push(delivered, DELIVERY, "delivery", (o, v))

        #largo de cola tras procesar el evento
        if len(queue) != qseries[-1][1]:
            qseries.append((t, len(queue)))

    recs = [records[i] for i in sorted(records)]
    served = [r for r in recs if r["estado"] == "entregado"]
    t_end = max(duration_min, last_return)
    nan = float("nan")

    #cola promedio en el tiempo
    area, (tp, qp) = 0.0, qseries[0]
    for t, q in qseries[1:]:
        area += qp * (t - tp)
        tp, qp = t, q
    area += qp * max(0.0, t_end - tp)

    n_on_time = sum(1 for r in served if r["puntual"])
    metrics = dict(
        pedidos=len(recs),
        entregados=len(served),
        sin_ruta=len(recs) - len(served),
        puntuales=n_on_time,
        pct_puntuales=100.0 * n_on_time / len(served) if served else nan,
        espera_media=float(np.mean([r["espera"] for r in served])) if served else nan,
        espera_max=max((r["espera"] for r in served), default=nan),
        respuesta_media=float(np.mean([r["solicitud_a_entrega"] for r in served])) if served else nan,
        utilizacion_pct=100.0 * busy / (n_vehicles * t_end) if t_end > 0 else 0.0,
        tiempo_adicional=max(0.0, last_delivery - duration_min),
        regreso_adicional=max(0.0, last_return - duration_min),
        cola_max=max(q for _, q in qseries),
        cola_media=area / t_end if t_end > 0 else 0.0,
    )
    return SimResult(recs, qseries, metrics, n_vehicles, duration_min)
