from __future__ import annotations
import math
import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from network import DEPOT, DESTINOS, Network, fmt_min

COLOR_POLICY = {"actual": "#4C78A8", "algoritmo": "#E8710A"}
POLICY_NAME = {"actual": "Rutas actuales", "algoritmo": "Rutas del algoritmo"}
DEST_COLORS = {"N08": "#1B9E77", "N19": "#7570B3", "N34": "#D81B60", "N40": "#0072B2"}

C_OK, C_MOD, C_CLOSED = "#9AA0A6", "#E8710A", "#D93025"
NODE_SIZE = 520
SHRINK = 13


def compute_layout(net: Network, seed: int = 7) -> dict:
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(net.nodes)
    for (u, v), w in net.original.items():
        w2 = min(w, net.original.get((v, u), w))
        if G.has_edge(u, v):
            G[u][v]["weight"] = min(G[u][v]["weight"], w2)
        else:
            G.add_edge(u, v, weight=w2)
    try:
        pos = nx.kamada_kawai_layout(G, weight="weight")
    except Exception:
        pos = nx.spring_layout(G, seed=seed)
    return {n: (float(x), float(y)) for n, (x, y) in pos.items()}


def _arc_mid(p1, p2, rad):
    (x1, y1), (x2, y2) = p1, p2
    dx, dy = x2 - x1, y2 - y1
    return (x1 + x2) / 2 + 0.5 * rad * dy, (y1 + y2) / 2 - 0.5 * rad * dx


def _arrow(ax, p1, p2, color, lw=1.0, style="-|>", rad=0.0, ls="-", alpha=1.0, z=2):
    ax.annotate("", xy=p2, xytext=p1, zorder=z, arrowprops=dict(
        arrowstyle=style, color=color, lw=lw, linestyle=ls, shrinkA=SHRINK, shrinkB=SHRINK,
        connectionstyle=f"arc3,rad={rad}", alpha=alpha, mutation_scale=9))


def draw_network(net: Network, pos: dict, highlight=None, figsize=(15, 10.5), show_weights=True) -> Figure:

    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    def cur(u, v):
        return net.time(u, v)

    def state(u, v):
        c, w0 = cur(u, v), net.original[(u, v)]
        return "closed" if c is None else ("ok" if c == w0 else "mod")

    colors = {"ok": C_OK, "mod": C_MOD, "closed": C_CLOSED}
    geom, drawn = {}, set()

    def label(p, text, color):
        if show_weights:
            ax.text(p[0], p[1], text, fontsize=6.2, ha="center", va="center", color=color, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))

    for (u, v) in net.original:
        key = frozenset((u, v))
        if key in drawn:
            continue
        drawn.add(key)
        p1, p2 = pos[u], pos[v]
        if (v, u) in net.original:
            s1, s2 = state(u, v), state(v, u)
            if s1 == s2 and cur(u, v) == cur(v, u):
                geom[(u, v)] = geom[(v, u)] = 0.0
                ls = "--" if s1 == "closed" else "-"
                _arrow(ax, p1, p2, colors[s1], 0.9, "<|-|>", 0.0, ls, 0.9)
                label(((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2),
                      "✕" if s1 == "closed" else fmt_min(cur(u, v)), colors[s1])
            else:
                for a, b in ((u, v), (v, u)):
                    geom[(a, b)] = 0.25
                    st = state(a, b)
                    _arrow(ax, pos[a], pos[b], colors[st], 0.9, "-|>", 0.25, "--" if st == "closed" else "-", 0.9)
                    label(_arc_mid(pos[a], pos[b], 0.25), "✕" if st == "closed" else fmt_min(cur(a, b)), colors[st])
        else:
            geom[(u, v)] = 0.0
            st = state(u, v)
            _arrow(ax, p1, p2, colors[st], 0.9, "-|>", 0.0, "--" if st == "closed" else "-", 0.9)
            label(((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2), "✕" if st == "closed" else fmt_min(cur(u, v)), colors[st])

    handles = []
    for h in highlight or []:
        path = h["path"]
        if not path:
            continue
        for a, b in zip(path, path[1:]):
            _arrow(ax, pos[a], pos[b], h["color"], h.get("lw", 3.0), "-|>", geom.get((a, b), 0.0),
                   h.get("ls", "-"), 0.95, z=3)
        handles.append(Line2D([0], [0], color=h["color"], lw=h.get("lw", 3.0), ls=h.get("ls", "-"),
                              label=h.get("label", "")))

    xs = [pos[n][0] for n in net.nodes]
    ys = [pos[n][1] for n in net.nodes]
    face = ["#2E9E5B" if n == DEPOT else ("#F4B400" if n in DESTINOS else "#F1F3F4") for n in net.nodes]
    ax.scatter(xs, ys, s=NODE_SIZE, c=face, edgecolors="#3C4043", linewidths=0.9, zorder=5)
    for n in net.nodes:
        ax.text(pos[n][0], pos[n][1], n, fontsize=6.4, ha="center", va="center", zorder=6,
                fontweight="bold", color="white" if n == DEPOT else "#202124")

    legend = [Patch(fc="#2E9E5B", ec="#3C4043", label="Bodega N01"),
              Patch(fc="#F4B400", ec="#3C4043", label="Zona de entrega"),
              Line2D([0], [0], color=C_OK, label="Tramo (minutos)"),
              Line2D([0], [0], color=C_MOD, label="Tiempo modificado"),
              Line2D([0], [0], color=C_CLOSED, ls="--", label="Tramo cerrado")] + handles
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    ax.set_title("Red vial: sentidos de circulación y tiempos de viaje (minutos)", fontsize=11)
    fig.tight_layout()
    return fig


#resultados
def _bar_axes(fig, axes, summary, metrics, titles, scenarios):
    width = 0.38
    x = np.arange(len(scenarios))
    for ax, m, title in zip(axes, metrics, titles):
        for i, p in enumerate(("actual", "algoritmo")):
            vals, errs = [], []
            for s in scenarios:
                r = summary[(summary.escenario == s) & (summary.politica == p) & (summary.metrica == m)].iloc[0]
                vals.append(r.media)
                errs.append(0.0 if math.isnan(r.ic95) else r.ic95)
            ax.bar(x + (i - 0.5) * width, vals, width, yerr=errs, capsize=3, color=COLOR_POLICY[p],
                   label=POLICY_NAME[p], error_kw=dict(lw=0.9, ecolor="#3C4043"))
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)


def plot_scenarios(summary, scenarios=None) -> Figure:
    scenarios = scenarios or list(dict.fromkeys(summary.escenario))
    metrics = ["pct_puntuales", "espera_media", "respuesta_media", "utilizacion_pct", "tiempo_adicional", "cola_max"]
    titles = ["Entregas puntuales (%)", "Espera media (min)", "Solicitud → entrega (min)",
              "Utilización de vehículos (%)", "Tiempo adicional para terminar (min)", "Cola máxima (pedidos)"]
    fig = Figure(figsize=(13, 7))
    axes = fig.subplots(2, 3).ravel()
    _bar_axes(fig, axes, summary, metrics, titles, scenarios)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Comparación de políticas por escenario (media e IC 95 %)", fontsize=11)
    fig.tight_layout()
    return fig


def plot_queue(results: dict, duration_min: float) -> Figure:
    fig = Figure(figsize=(9, 3.6))
    ax = fig.add_subplot(111)
    for p, res in results.items():
        t = [a for a, _ in res.queue_series]
        q = [b for _, b in res.queue_series]
        ax.step(t, q, where="post", color=COLOR_POLICY[p], label=POLICY_NAME[p], lw=1.6)
    ax.axvline(duration_min, color="#80868B", ls=":", lw=1)
    ax.text(duration_min, ax.get_ylim()[1] * 0.97, " cierre de recepción", fontsize=7, color="#5F6368", va="top")
    ax.set_xlabel("Minutos desde el inicio")
    ax.set_ylabel("Pedidos en cola")
    ax.set_title("Cola de pedidos esperando vehículo", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_sweep(sweep, rate) -> Figure:
    fig = Figure(figsize=(12, 3.8))
    axes = fig.subplots(1, 3)
    specs = [("pct_puntuales", "Entregas puntuales (%)"), ("espera_media", "Espera media (min)"),
             ("utilizacion_pct", "Utilización (%)")]
    for ax, (m, title) in zip(axes, specs):
        for p in ("actual", "algoritmo"):
            d = sweep[sweep.politica == p].sort_values("vehiculos")
            ic = d[f"{m}_ic95"].fillna(0.0)
            ax.errorbar(d.vehiculos, d[f"{m}_media"], yerr=ic, marker="o", ms=4, lw=1.5, capsize=2.5,
                        color=COLOR_POLICY[p], label=POLICY_NAME[p])
        ax.set_xlabel("Vehículos")
        ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Tamaño de flota con {rate:g} pedidos/hora", fontsize=10)
    fig.tight_layout()
    return fig
