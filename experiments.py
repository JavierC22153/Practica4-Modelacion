from __future__ import annotations

import math
import random
import time

import numpy as np
import pandas as pd
from scipy import stats

from network import DEPOT, DESTINOS, RUTAS_ACTUALES, Network, fmt_min
from routing import (ALT, FloydWarshall, astar, bellman_ford, dijkstra, plan_routes,
                     route_options)
from simulation import PUNTUAL_MIN, generate_orders, simulate

SCENARIOS = {"Normal": 4.0, "Semana Santa": 6.0, "Navidad": 10.0}
POLICIES = ("actual", "algoritmo")
POLICY_LABEL = {"actual": "Rutas actuales", "algoritmo": "Rutas del algoritmo"}

METRIC_LABELS = {
    "pedidos": "Pedidos recibidos",
    "puntuales": "Entregas puntuales",
    "pct_puntuales": "% puntuales",
    "espera_media": "Espera media (min)",
    "respuesta_media": "Solicitud → entrega (min)",
    "utilizacion_pct": "Utilización de vehículos (%)",
    "tiempo_adicional": "Tiempo adicional (min)",
    "cola_max": "Cola máxima",
}
KEY_METRICS = list(METRIC_LABELS)


def _path_str(path):
    return " → ".join(path) if path else "—"


# rutas / grafo
def validation_table(net: Network) -> pd.DataFrame:
    rows = []
    for d in DESTINOS:
        hoja = RUTAS_ACTUALES[d]
        chk = net.check_route(hoja, DEPOT, d)
        rows.append({
            "Destino": d,
            "Ruta (ida)": _path_str(hoja),
            "Ida válida": chk["ida"].valid,
            "Ida (min)": chk["ida"].minutes,
            "Regreso por camino inverso válido": chk["regreso"].valid,
            "Regreso (min)": chk["regreso"].minutes,
            "Tramos faltantes (ida)": ", ".join(f"{a}→{b}" for a, b in chk["ida"].missing) or "—",
            "Tramos faltantes (regreso)": ", ".join(f"{a}→{b}" for a, b in chk["regreso"].missing) or "—",
        })
    return pd.DataFrame(rows)


def route_comparison_table(net: Network, n_alt=2, service_min=0.0) -> pd.DataFrame:
    rows = []
    for d in DESTINOS:
        opts = route_options(net, d, DEPOT, n_alt)
        base = opts[0].round_trip
        for o in opts:
            rows.append({
                "Destino": d,
                "Ruta": o.label,
                "Camino de ida": _path_str(o.out_path),
                "Ida (min)": o.out_min,
                "Camino de regreso": _path_str(o.back_path),
                "Regreso (min)": o.back_min,
                "Ida + regreso (min)": o.round_trip,
                "Δ vs actual (min)": None if o.round_trip is None or base is None else o.round_trip - base,
                f"Ida ≤ {PUNTUAL_MIN:g} min sin espera": (o.out_min is not None
                                                          and o.out_min + service_min <= PUNTUAL_MIN),
                "Nota": o.note,
            })
    return pd.DataFrame(rows)


def benchmark_algorithms(net: Network, n_random=100, seed=1) -> pd.DataFrame:
    rng = random.Random(seed)
    queries = [(DEPOT, d) for d in DESTINOS] + [(d, DEPOT) for d in DESTINOS]
    nodes = net.nodes
    while len(queries) < 8 + n_random:
        a, b = rng.sample(nodes, 2)
        if dijkstra(net, a, b) is not None:
            queries.append((a, b))

    ref = {q: dijkstra(net, *q).cost for q in queries}
    rows = []

    def run(name, fn, prep=None):
        t0 = time.perf_counter()
        ctx = prep() if prep else None
        costs, work = [], []
        for q in queries:
            res = fn(ctx, *q)
            costs.append(res.cost)
            work.append(res.expanded)
        ms = (time.perf_counter() - t0) * 1000
        err = max(abs(c - ref[q]) for c, q in zip(costs, queries))
        rows.append({"Algoritmo": name, "Consultas": len(queries),
                     "Nodos expandidos / consulta (prom.)": float(np.mean(work)),
                     "Tiempo total (ms)": ms, "Diferencia máx. de costo vs Dijkstra": err})

    run("Dijkstra", lambda ctx, a, b: dijkstra(net, a, b))
    run("A* con heurística ALT (incluye precálculo)", lambda ctx, a, b: astar(net, a, b, ctx), prep=lambda: ALT(net))
    run("Bellman-Ford (relajaciones)", lambda ctx, a, b: bellman_ford(net, a, b))

    t0 = time.perf_counter()
    fw = FloydWarshall(net)
    costs = [fw.route(a, b).cost for a, b in queries]
    ms = (time.perf_counter() - t0) * 1000
    rows.append({"Algoritmo": "Floyd-Warshall (n³ operaciones, una vez)", "Consultas": len(queries),
                 "Nodos expandidos / consulta (prom.)": float(fw.operations) / len(queries),
                 "Tiempo total (ms)": ms,
                 "Diferencia máx. de costo vs Dijkstra": max(abs(c - ref[q]) for c, q in zip(costs, queries))})
    return pd.DataFrame(rows)


#estadística
def ci95(values) -> float:
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))


def _run_reps(net, plans, rate, n_reps, n_vehicles, duration_h, base_seed, service_min, extra=None):
    rows = []
    for rep in range(n_reps):
        seed = base_seed + rep
        orders = generate_orders(rate, duration_h, seed)       # mismos pedidos para ambas políticas
        for policy in POLICIES:
            res = simulate(orders, plans[policy], n_vehicles, duration_h * 60.0, service_min)
            row = {"rep": rep + 1, "seed": seed, "politica": policy, **res.metrics}
            if extra:
                row.update(extra)
            rows.append(row)
    return rows


def run_experiments(net: Network, scenarios=None, n_reps=10, n_vehicles=8, duration_h=8.0,
                    base_seed=2026, service_min=0.0) -> dict:
    scenarios = scenarios or SCENARIOS
    plans = {p: plan_routes(net, p) for p in POLICIES}
    rows = []
    for name, rate in scenarios.items():
        rows += _run_reps(net, plans, rate, n_reps, n_vehicles, duration_h, base_seed, service_min,
                          extra={"escenario": name, "tasa": rate})
    runs = pd.DataFrame(rows)

    summ, paired = [], []
    for name in scenarios:
        sub = runs[runs.escenario == name]
        for m in KEY_METRICS:
            for p in POLICIES:
                v = sub[sub.politica == p][m]
                summ.append({"escenario": name, "politica": p, "metrica": m,
                             "media": v.mean(), "ic95": ci95(v), "n": len(v)})
            a = sub[sub.politica == "algoritmo"].sort_values("rep")[m].to_numpy()
            b = sub[sub.politica == "actual"].sort_values("rep")[m].to_numpy()
            diff = a - b
            paired.append({"escenario": name, "metrica": m, "dif_media": float(np.nanmean(diff)),
                           "ic95": ci95(diff), "n": len(diff)})
    return {"runs": runs, "summary": pd.DataFrame(summ), "paired": pd.DataFrame(paired)}


def summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (esc, pol), g in summary.groupby(["escenario", "politica"], sort=False):
        row = {"Escenario": esc, "Política": POLICY_LABEL[pol]}
        for _, r in g.iterrows():
            ic = "" if math.isnan(r.ic95) else f" ± {r.ic95:.1f}"
            row[METRIC_LABELS[r.metrica]] = f"{r.media:.1f}{ic}"
        rows.append(row)
    return pd.DataFrame(rows)


def paired_table(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for esc, g in paired.groupby("escenario", sort=False):
        row = {"Escenario": esc}
        for _, r in g.iterrows():
            ic = "" if math.isnan(r.ic95) else f" ± {r.ic95:.1f}"
            row[METRIC_LABELS[r.metrica]] = f"{r.dif_media:+.1f}{ic}"
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_vehicles(net: Network, rate, vehicle_values, n_reps=10, duration_h=8.0, base_seed=2026,
                   service_min=0.0) -> pd.DataFrame:
    plans = {p: plan_routes(net, p) for p in POLICIES}
    rows = []
    for v in vehicle_values:
        rows += _run_reps(net, plans, rate, n_reps, v, duration_h, base_seed, service_min,
                          extra={"vehiculos": v})
    runs = pd.DataFrame(rows)
    keep = ["pct_puntuales", "espera_media", "respuesta_media", "utilizacion_pct", "tiempo_adicional", "cola_max"]
    g = runs.groupby(["vehiculos", "politica"])[keep]
    mean = g.mean().add_suffix("_media")
    ic = g.agg(ci95).add_suffix("_ic95")
    return pd.concat([mean, ic], axis=1).reset_index()


def capacity_estimate(net: Network, service_min=0.0) -> pd.DataFrame:
    rows = []
    for p in POLICIES:
        plans = plan_routes(net, p)
        trips = [pl.out_min + service_min + pl.back_min for pl in plans.values() if pl.reachable]
        mean_trip = float(np.mean(trips))
        rows.append({"Política": POLICY_LABEL[p], "Duración media de un viaje redondo (min)": mean_trip,
                     "Pedidos/hora por vehículo": 60.0 / mean_trip})
    return pd.DataFrame(rows)
