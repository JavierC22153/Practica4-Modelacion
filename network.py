from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, field
from pathlib import Path

DEPOT = "N01"
DESTINOS = ["N08", "N19", "N34", "N40"]

# Rutas de las hojas de despacho.
RUTAS_ACTUALES = {
    "N08": ["N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08"],
    "N19": ["N01", "N02", "N03", "N11", "N19"],
    "N34": ["N01", "N09", "N17", "N25", "N33", "N34"],
    "N40": ["N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08",
            "N16", "N24", "N32", "N40"],
}

_SIN_CONEXION = {"", "—", "–", "-"}

DEFAULT_CSV = Path(__file__).with_name("matriz.csv")


@dataclass
class PathCheck:
    valid: bool
    minutes: float | None          # None si el camino no es válido
    steps: list = field(default_factory=list)    # [(u, v, minutos)] de los tramos que sí existen
    missing: list = field(default_factory=list)  # [(u, v)] tramos inexistentes o cerrados


class Network:
    def __init__(self, nodes, edges):
        self.nodes = list(nodes)
        self.adj = {n: {} for n in self.nodes}
        for (u, v), w in edges.items():
            self.adj[u][v] = float(w)
        self.original = {(u, v): w for u in self.adj for v, w in self.adj[u].items()}
        self.version = 0  # aumenta con cada incidente; invalida cachés de rutas

    #carga
    @classmethod
    def from_csv(cls, path=DEFAULT_CSV):
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        cols = [c.strip() for c in rows[0][1:]]
        edges, row_names = {}, []
        for r in rows[1:]:
            if not r or not r[0].strip():
                continue
            u = r[0].strip()
            row_names.append(u)
            for v, cell in zip(cols, r[1:]):
                cell = cell.strip()
                if cell in _SIN_CONEXION:
                    continue
                w = float(cell)
                if u == v:
                    if w != 0:
                        raise ValueError(f"La diagonal debe ser 0 ({u}: {w})")
                    continue
                if w <= 0:
                    raise ValueError(f"Tiempo no positivo en {u}->{v}: {w}")
                edges[(u, v)] = w
        if row_names != cols:
            raise ValueError("Las filas y columnas de la matriz no coinciden")
        return cls(cols, edges)

    def copy(self) -> "Network":
        return copy.deepcopy(self)

    #consulta
    def time(self, u, v):
        return self.adj.get(u, {}).get(v)

    def edges(self):
        return [(u, v, w) for u in self.adj for v, w in self.adj[u].items()]

    def reverse_adj(self):
        radj = {n: {} for n in self.nodes}
        for u in self.adj:
            for v, w in self.adj[u].items():
                radj[v][u] = w
        return radj

    def asymmetric_pairs(self):
        out = []
        for (u, v), w in self.original.items():
            if u < v:
                back = self.original.get((v, u))
                if back is None or back != w:
                    out.append((u, v, w, back))
            elif (v, u) not in self.original:
                out.append((u, v, w, None))
        return out

    #validación
    def check_path(self, path) -> PathCheck:
        steps, missing, total = [], [], 0.0
        for u, v in zip(path, path[1:]):
            w = self.time(u, v)
            if w is None:
                missing.append((u, v))
            else:
                steps.append((u, v, w))
                total += w
        valid = not missing and len(path) >= 1
        return PathCheck(valid, total if valid else None, steps, missing)

    def path_cost(self, path) -> float:
        chk = self.check_path(path)
        if not chk.valid:
            raise ValueError(f"Camino inválido, faltan tramos: {chk.missing}")
        return chk.minutes

    def check_route(self, path, depot=DEPOT, dest=None) -> dict:
        dest = dest or path[-1]
        ida = self.check_path(path)
        regreso = self.check_path(list(reversed(path)))
        endpoints = path[0] == depot and path[-1] == dest
        simple = len(set(path)) == len(path)
        return {
            "ida": ida,
            "regreso": regreso,
            "extremos_ok": endpoints,
            "sin_repetidos": simple,
            "valida": ida.valid and regreso.valid and endpoints and simple,
        }

    #incidentes
    def _touch(self):
        self.version += 1

    def _targets(self, u, v, both):
        pairs = [(u, v)] + ([(v, u)] if both else [])
        pairs = [p for p in pairs if p in self.original]
        if not pairs:
            raise KeyError(f"No existe el tramo {u}->{v} en la red original")
        return pairs

    def set_time(self, u, v, minutes, both=False):
        if minutes <= 0:
            raise ValueError("El tiempo debe ser positivo; para bloquear use close()")
        for a, b in self._targets(u, v, both):
            self.adj[a][b] = float(minutes)
        self._touch()

    def close(self, u, v, both=False):
        for a, b in self._targets(u, v, both):
            self.adj[a].pop(b, None)
        self._touch()

    def restore(self, u, v, both=False):
        for a, b in self._targets(u, v, both):
            self.adj[a][b] = self.original[(a, b)]
        self._touch()

    def reset(self):
        for u in self.adj:
            self.adj[u] = {}
        for (u, v), w in self.original.items():
            self.adj[u][v] = w
        self._touch()

    def changes(self):
        out = []
        for (u, v), w0 in self.original.items():
            cur = self.adj[u].get(v)
            if cur != w0:
                out.append({"origen": u, "destino": v, "original": w0, "actual": cur})
        return out


def fmt_min(x) -> str:
    if x is None:
        return "—"
    return f"{x:g}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"
