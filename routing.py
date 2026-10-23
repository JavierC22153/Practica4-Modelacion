from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from network import DEPOT, DESTINOS, RUTAS_ACTUALES, Network

INF = math.inf


@dataclass
class RouteResult:
    cost: float
    path: list
    expanded: int = 0   # nodos extraídos de la cola


def _reconstruct(prev, src, dst):
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return path


#Dijkstra
def dijkstra(net: Network, src, dst, banned_edges=frozenset(), banned_nodes=frozenset()):
    if src in banned_nodes or dst in banned_nodes:
        return None
    dist = {src: 0.0}
    prev = {}
    done = set()
    heap = [(0.0, src)]
    expanded = 0
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        expanded += 1
        if u == dst:
            return RouteResult(d, _reconstruct(prev, src, dst), expanded)
        for v, w in net.adj[u].items():
            if v in done or v in banned_nodes or (u, v) in banned_edges:
                continue
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return None


def _sssp(adj, src):
    dist = {src: 0.0}
    heap = [(0.0, src)]
    done = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


#A* con ALT
class ALT:
    def __init__(self, net: Network, n_landmarks: int = 4):
        self.version = net.version
        radj = net.reverse_adj()
        first_dist = _sssp(net.adj, net.nodes[0])
        first = max(first_dist, key=first_dist.get)
        self.landmarks = [first]
        self.d_from = {first: _sssp(net.adj, first)}
        self.d_to = {first: _sssp(radj, first)}
        while len(self.landmarks) < min(n_landmarks, len(net.nodes)):
            best, best_score = None, -1.0
            for v in net.nodes:
                if v in self.landmarks:
                    continue
                scores = []
                for L in self.landmarks:
                    if v in self.d_from[L] and v in self.d_to[L]:
                        scores.append(self.d_from[L][v] + self.d_to[L][v])
                if scores and min(scores) > best_score:
                    best, best_score = v, min(scores)
            if best is None:
                break
            self.landmarks.append(best)
            self.d_from[best] = _sssp(net.adj, best)
            self.d_to[best] = _sssp(radj, best)

    def __call__(self, v, t):
        best = 0.0
        for L in self.landmarks:
            a, b = self.d_from[L], self.d_to[L]
            if v in a and t in a:
                best = max(best, a[t] - a[v])
            if v in b and t in b:
                best = max(best, b[v] - b[t])
        return best


def astar(net: Network, src, dst, h=None):
    if h is None:
        h = ALT(net)
    g = {src: 0.0}
    prev = {}
    done = set()
    heap = [(h(src, dst), src)]
    expanded = 0
    while heap:
        _, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        expanded += 1
        if u == dst:
            return RouteResult(g[u], _reconstruct(prev, src, dst), expanded)
        for v, w in net.adj[u].items():
            if v in done:
                continue
            ng = g[u] + w
            if ng < g.get(v, INF):
                g[v] = ng
                prev[v] = u
                heapq.heappush(heap, (ng + h(v, dst), v))
    return None


#Bellman-Ford
def bellman_ford(net: Network, src, dst):
    dist = {n: INF for n in net.nodes}
    prev = {}
    dist[src] = 0.0
    edges = net.edges()
    checks = 0
    for _ in range(len(net.nodes) - 1):
        changed = False
        for u, v, w in edges:
            checks += 1
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                changed = True
        if not changed:
            break
    for u, v, w in edges:  # detección de ciclos negativos
        if dist[u] + w < dist[v]:
            raise ValueError("Ciclo de peso negativo")
    if dist[dst] == INF:
        return None
    return RouteResult(dist[dst], _reconstruct(prev, src, dst), checks)


#Floyd-Warshall
class FloydWarshall:
    def __init__(self, net: Network):
        self.nodes = net.nodes
        n = len(self.nodes)
        idx = {v: i for i, v in enumerate(self.nodes)}
        self.idx = idx
        self.dist = [[INF] * n for _ in range(n)]
        self.nxt = [[None] * n for _ in range(n)]
        for i in range(n):
            self.dist[i][i] = 0.0
            self.nxt[i][i] = i
        for u, v, w in net.edges():
            i, j = idx[u], idx[v]
            self.dist[i][j] = w
            self.nxt[i][j] = j
        self.operations = n ** 3
        d, nx = self.dist, self.nxt
        for k in range(n):
            dk = d[k]
            for i in range(n):
                dik = d[i][k]
                if dik == INF:
                    continue
                di = d[i]
                for j in range(n):
                    nd = dik + dk[j]
                    if nd < di[j]:
                        di[j] = nd
                        nx[i][j] = nx[i][k]

    def route(self, src, dst):
        i, j = self.idx[src], self.idx[dst]
        if self.dist[i][j] == INF:
            return None
        path = [src]
        while i != j:
            i = self.nxt[i][j]
            path.append(self.nodes[i])
        return RouteResult(self.dist[self.idx[src]][j], path, 0)


#Yen
def k_shortest_paths(net: Network, src, dst, k=3):
    first = dijkstra(net, src, dst)
    if first is None:
        return []
    A = [first]
    B = []
    seen = {tuple(first.path)}
    for _ in range(1, k):
        last = A[-1].path
        for i in range(len(last) - 1):
            spur = last[i]
            root = last[: i + 1]
            banned_edges = set()
            for p in A:
                if p.path[: i + 1] == root and len(p.path) > i + 1:
                    banned_edges.add((p.path[i], p.path[i + 1]))
            res = dijkstra(net, spur, dst, banned_edges, set(root[:-1]))
            if res is not None:
                full = tuple(root[:-1] + res.path)
                if full not in seen:
                    seen.add(full)
                    heapq.heappush(B, (net.path_cost(list(full)), full))
        if not B:
            break
        cost, p = heapq.heappop(B)
        A.append(RouteResult(cost, list(p), 0))
    return A


#planes de ruta (viaje)
@dataclass
class RoutePlan:
    dest: str
    label: str
    out_path: list | None
    out_min: float | None
    back_path: list | None
    back_min: float | None
    note: str = ""

    @property
    def reachable(self) -> bool:
        return self.out_min is not None and self.back_min is not None

    @property
    def round_trip(self):
        return None if not self.reachable else self.out_min + self.back_min


def _plan_from_paths(net, dest, label, out_path, back_path, note=""):
    out_min = net.path_cost(out_path) if out_path else None
    back_min = net.path_cost(back_path) if back_path else None
    return RoutePlan(dest, label, out_path, out_min, back_path, back_min, note)


def _shortest(net, engine, src, dst):
    res = astar(net, src, dst, engine)
    return res.path if res else None


def plan_routes(net: Network, policy: str, depot=DEPOT, destinos=DESTINOS) -> dict:
    engine = ALT(net)
    plans = {}
    for d in destinos:
        if policy == "actual":
            hoja = RUTAS_ACTUALES.get(d)
            notes = []
            out = hoja if hoja and net.check_path(hoja).valid else None
            if out is None:
                out = _shortest(net, engine, depot, d)
                notes.append("ida: hoja bloqueada, desvío por ruta más corta")
            back = list(reversed(hoja)) if hoja and net.check_path(list(reversed(hoja))).valid else None
            if back is None:
                back = _shortest(net, engine, d, depot)
                notes.append("regreso: camino inverso no disponible, ruta más corta")
            plans[d] = _plan_from_paths(net, d, "Actual (hoja)", out, back, "; ".join(notes))
        elif policy == "algoritmo":
            plans[d] = _plan_from_paths(net, d, "Algoritmo (A* ALT)",
                                        _shortest(net, engine, depot, d),
                                        _shortest(net, engine, d, depot))
        else:
            raise ValueError(f"Política desconocida: {policy}")
    return plans


def route_options(net: Network, dest, depot=DEPOT, n_alt=2) -> list:
    actual = plan_routes(net, "actual", depot, [dest])[dest]
    algo = plan_routes(net, "algoritmo", depot, [dest])[dest]
    options = [actual, algo]
    excluded = {tuple(actual.out_path or []), tuple(algo.out_path or [])}
    engine = ALT(net)
    n = 0
    for res in k_shortest_paths(net, depot, dest, k=n_alt + 4):
        if tuple(res.path) in excluded:
            continue
        n += 1
        rev = list(reversed(res.path))
        if net.check_path(rev).valid:
            back, note = rev, "regreso por el camino inverso"
        else:
            back, note = _shortest(net, engine, dest, depot), "regreso por la ruta más corta (inverso no válido)"
        options.append(_plan_from_paths(net, dest, f"Alternativa {n}", res.path, back, note))
        if n >= n_alt:
            break
    return options
