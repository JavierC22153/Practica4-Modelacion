import argparse
from pathlib import Path

import pandas as pd

from experiments import (SCENARIOS, benchmark_algorithms, capacity_estimate, paired_table,
                         route_comparison_table, run_experiments, summary_table, sweep_vehicles,
                         validation_table)
from network import DEFAULT_CSV, DESTINOS, Network
from plots import (DEST_COLORS, compute_layout, draw_network, plot_queue, plot_scenarios, plot_sweep)
from routing import plan_routes
from simulation import generate_orders, simulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--reps", type=int, default=30, help="repeticiones por escenario (mínimo pedido: 5)")
    ap.add_argument("--vehicles", type=int, default=8)
    ap.add_argument("--duration", type=float, default=8.0, help="horas de recepción de pedidos")
    ap.add_argument("--service", type=float, default=0.0, help="minutos de servicio en el destino")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="resultados")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    net = Network.from_csv(a.csv)
    pos = compute_layout(net)

    # Grafo con las rutas actuales (discontinuas) y las del algoritmo (continuas)
    plans = {p: plan_routes(net, p) for p in ("actual", "algoritmo")}
    hl = []
    for d in DESTINOS:
        hl.append(dict(path=plans["actual"][d].out_path, color=DEST_COLORS[d], lw=4.0, ls=":", label=f"Actual → {d}"))
    for d in DESTINOS:
        hl.append(dict(path=plans["algoritmo"][d].out_path, color=DEST_COLORS[d], lw=2.2, ls="-", label=f"Algoritmo → {d}"))
    draw_network(net, pos).savefig(out / "grafo.png", dpi=170)
    draw_network(net, pos, highlight=hl).savefig(out / "grafo_rutas.png", dpi=170)

    validation_table(net).to_csv(out / "validacion_rutas.csv", index=False)
    route_comparison_table(net, service_min=a.service).to_csv(out / "tabla_rutas.csv", index=False)
    benchmark_algorithms(net).to_csv(out / "algoritmos.csv", index=False)
    capacity_estimate(net, a.service).to_csv(out / "capacidad.csv", index=False)

    res = run_experiments(net, SCENARIOS, a.reps, a.vehicles, a.duration, a.seed, a.service)
    res["runs"].to_csv(out / "corridas.csv", index=False)
    res["summary"].to_csv(out / "resumen_largo.csv", index=False)
    summary_table(res["summary"]).to_csv(out / "resumen_escenarios.csv", index=False)
    paired_table(res["paired"]).to_csv(out / "diferencias_pareadas.csv", index=False)
    plot_scenarios(res["summary"]).savefig(out / "escenarios.png", dpi=170)

    #cola en el tiempo
    for name, rate in SCENARIOS.items():
        orders = generate_orders(rate, a.duration, a.seed)
        results = {p: simulate(orders, plans[p], a.vehicles, a.duration * 60, a.service) for p in plans}
        plot_queue(results, a.duration * 60).savefig(out / f"cola_{name.lower().replace(' ', '_')}.png", dpi=170)

    #Tamaño de flota para cada escenario
    for name, rate in SCENARIOS.items():
        sw = sweep_vehicles(net, rate, range(2, 21, 2), a.reps, a.duration, a.seed, a.service)
        sw.to_csv(out / f"flota_{name.lower().replace(' ', '_')}.csv", index=False)
        plot_sweep(sw, rate).savefig(out / f"flota_{name.lower().replace(' ', '_')}.png", dpi=170)

    print(f"Listo: resultados en {out.resolve()}")
    print(summary_table(res["summary"]).to_string(index=False))
    print(paired_table(res["paired"]).to_string(index=False))


if __name__ == "__main__":
    main()
