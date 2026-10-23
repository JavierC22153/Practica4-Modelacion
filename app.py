import pandas as pd
import streamlit as st

from experiments import (SCENARIOS, METRIC_LABELS, POLICIES, POLICY_LABEL, benchmark_algorithms,
                         capacity_estimate, paired_table, route_comparison_table, run_experiments,
                         summary_table, sweep_vehicles, validation_table)
from network import DEFAULT_CSV, DESTINOS, Network, fmt_min
from plots import DEST_COLORS, compute_layout, draw_network, plot_queue, plot_scenarios, plot_sweep
from routing import plan_routes
from simulation import PUNTUAL_MIN, generate_orders, simulate

st.set_page_config(page_title="Logística del Valle", layout="wide")


#estado
@st.cache_resource
def _load_original():
    net = Network.from_csv(DEFAULT_CSV)
    return net, compute_layout(net)


original, layout = _load_original()
if "net" not in st.session_state:
    st.session_state.net = original.copy()
    st.session_state.setdefault("scenario", "Normal")
    st.session_state.setdefault("rate", 4.0)
net: Network = st.session_state.net


def _on_scenario():
    sc = st.session_state.scenario
    if sc in SCENARIOS:
        st.session_state.rate = float(SCENARIOS[sc])


#barra lateral
st.sidebar.title("Parámetros")
st.sidebar.selectbox("Escenario", list(SCENARIOS) + ["Personalizado"], key="scenario", on_change=_on_scenario)
st.sidebar.number_input("Pedidos por hora (promedio)", min_value=0.5, max_value=60.0, step=0.5, key="rate")
n_vehicles = st.sidebar.number_input("Vehículos", min_value=1, max_value=60, value=8, step=1)
duration_h = st.sidebar.number_input("Duración de la recepción de pedidos (horas)", min_value=1.0, max_value=24.0,
                                     value=8.0, step=0.5)
seed = st.sidebar.number_input("Semilla aleatoria", min_value=0, value=2026, step=1)
service_min = st.sidebar.number_input("Tiempo de entrega en el destino (min)", min_value=0.0, max_value=60.0,
                                      value=0.0, step=1.0)
n_reps = st.sidebar.number_input("Repeticiones por escenario (experimentos)", min_value=5, max_value=200,
                                 value=10, step=1)
rate = float(st.session_state.rate)

with st.sidebar.expander("Incidente vial", expanded=bool(net.changes())):
    edges = sorted(original.original.items())
    labels = {f"{u} → {v}": (u, v) for (u, v), _ in edges}
    pick = st.selectbox("Tramo", list(labels),
                        format_func=lambda k: f"{k}  ({fmt_min(original.original[labels[k]])} min orig., "
                                              f"hoy {fmt_min(net.time(*labels[k]))})")
    action = st.radio("Acción", ["Cambiar tiempo", "Cerrar tramo", "Restaurar tramo"], horizontal=False)
    u, v = labels[pick]
    new_t = st.number_input("Nuevo tiempo (min)", min_value=1.0, max_value=600.0,
                            value=float(original.original[(u, v)]), step=1.0,
                            disabled=action != "Cambiar tiempo")
    both = st.checkbox("Aplicar también al sentido contrario (si existe)", value=False)
    c1, c2 = st.columns(2)
    if c1.button("Aplicar", type="primary"):
        if action == "Cambiar tiempo":
            net.set_time(u, v, new_t, both)
        elif action == "Cerrar tramo":
            net.close(u, v, both)
        else:
            net.restore(u, v, both)
        st.rerun()
    if c2.button("Restablecer red"):
        net.reset()
        st.rerun()
    ch = net.changes()
    if ch:
        st.caption("Cambios activos respecto a la red original:")
        st.dataframe(pd.DataFrame(ch).assign(actual=lambda d: d.actual.map(lambda x: "cerrado" if pd.isna(x) else x)),
                     hide_index=True)
    st.caption("Al cambiar la red, las rutas del algoritmo se recalculan (se reconstruye la heurística ALT). "
               "Las hojas actuales se mantienen mientras todos sus tramos existan; si uno se cierra, el "
               "conductor se desvía por el camino más corto.")

#contenido
st.title("Logística del Valle — Rutas y flota")
st.caption(f"Red con {len(net.nodes)} puntos y {len(net.edges())} tramos dirigidos · versión de red {net.version} · "
           f"{rate:g} pedidos/h · {n_vehicles} vehículos · {duration_h:g} h de recepción")

plans = {p: plan_routes(net, p) for p in POLICIES}
tab_graph, tab_routes, tab_sim, tab_exp, tab_fleet = st.tabs(
    ["Grafo", "Rutas", "Simulación", "Experimentos", "Flota"])

#Grafo
with tab_graph:
    sel = st.multiselect("Resaltar rutas de ida a", DESTINOS, default=DESTINOS)
    show_w = st.checkbox("Mostrar tiempos en los tramos", value=True)
    hl = []
    for d in sel:
        hl.append(dict(path=plans["actual"][d].out_path, color=DEST_COLORS[d], lw=4.0, ls=":", label=f"Actual → {d}"))
    for d in sel:
        hl.append(dict(path=plans["algoritmo"][d].out_path, color=DEST_COLORS[d], lw=2.2, ls="-",
                       label=f"Algoritmo → {d}"))
    st.pyplot(draw_network(net, layout, highlight=hl, show_weights=show_w))
    st.caption("La disposición es ilustrativa (no hay mapa): se calculó con los tiempos como distancias. "
               "Punteado grueso = ruta actual; línea continua = ruta del algoritmo.")

#Rutas
with tab_routes:
    st.subheader("Validez de las hojas de despacho")
    st.dataframe(validation_table(net), hide_index=True)
    st.subheader("Comparación de rutas por destino")
    st.dataframe(route_comparison_table(net, n_alt=2, service_min=service_min), hide_index=True)
    st.subheader("Capacidad teórica de la flota")
    cap = capacity_estimate(net, service_min)
    cap["Pedidos/hora que sostiene la flota"] = cap["Pedidos/hora por vehículo"] * n_vehicles
    st.dataframe(cap, hide_index=True)
    with st.expander("Comparación de algoritmos sobre esta red"):
        st.dataframe(benchmark_algorithms(net), hide_index=True)

#Simulación
with tab_sim:
    st.write(f"Una jornada con **{rate:g} pedidos/h**, semilla **{seed}**. Ambas políticas reciben "
             "exactamente los mismos pedidos, destinos e instantes de llegada.")
    if st.button("Ejecutar simulación", type="primary"):
        orders = generate_orders(rate, duration_h, int(seed))
        st.session_state.sim = (orders, {p: simulate(orders, plans[p], int(n_vehicles), duration_h * 60,
                                                     service_min) for p in POLICIES}, duration_h * 60)
    if "sim" in st.session_state:
        orders, results, dur = st.session_state.sim
        rows = []
        for p in POLICIES:
            m = results[p].metrics
            rows.append({"Política": POLICY_LABEL[p], "Pedidos": m["pedidos"],
                         f"Puntuales (≤ {PUNTUAL_MIN:g} min)": f'{m["puntuales"]} ({m["pct_puntuales"]:.0f} %)',
                         "Espera media (min)": round(m["espera_media"], 1),
                         "Solicitud → entrega (min)": round(m["respuesta_media"], 1),
                         "Utilización (%)": round(m["utilizacion_pct"], 1),
                         "Tiempo adicional (min)": round(m["tiempo_adicional"], 1),
                         "Cola máx.": m["cola_max"], "Sin ruta": m["sin_ruta"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.pyplot(plot_queue(results, dur))
        pol = st.radio("Detalle de pedidos de", POLICIES, format_func=POLICY_LABEL.get, horizontal=True)
        df = pd.DataFrame(results[pol].records)
        st.dataframe(df, hide_index=True)
        st.download_button("Descargar pedidos (CSV)", df.to_csv(index=False).encode("utf-8"),
                           file_name=f"pedidos_{pol}.csv", mime="text/csv")

#Experimentos
with tab_exp:
    st.write(f"{n_reps} repeticiones por escenario con semillas consecutivas desde {seed}, "
             f"{n_vehicles} vehículos, red actual (incluye incidentes activos). En cada repetición ambas "
             "políticas usan los mismos pedidos.")
    if st.button("Ejecutar experimentos", type="primary"):
        st.session_state.exp = run_experiments(net, SCENARIOS, int(n_reps), int(n_vehicles), duration_h,
                                               int(seed), service_min)
    if "exp" in st.session_state:
        exp = st.session_state.exp
        st.subheader("Media ± IC 95 %")
        st.dataframe(summary_table(exp["summary"]), hide_index=True)
        st.subheader("Diferencia pareada (algoritmo − actual)")
        st.dataframe(paired_table(exp["paired"]), hide_index=True)
        st.pyplot(plot_scenarios(exp["summary"]))
        st.download_button("Descargar corridas (CSV)", exp["runs"].to_csv(index=False).encode("utf-8"),
                           file_name="corridas.csv", mime="text/csv")

#Flota
with tab_fleet:
    st.write(f"Efecto del número de vehículos con **{rate:g} pedidos/h**.")
    lo, hi = st.slider("Rango de vehículos", 1, 40, (2, 20))
    if st.button("Calcular", type="primary"):
        st.session_state.sweep = (rate, sweep_vehicles(net, rate, range(lo, hi + 1), int(n_reps), duration_h,
                                                       int(seed), service_min))
    if "sweep" in st.session_state:
        r, sw = st.session_state.sweep
        st.pyplot(plot_sweep(sw, r))
        st.dataframe(sw.round(2), hide_index=True)
