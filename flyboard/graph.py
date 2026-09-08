"""Graph loading: full MaleCNS net, LITE BRAIN subsample, or synthetic fixture."""
from __future__ import annotations

import json
import os

import numpy as np
import scipy.sparse as sp

from . import FOOTER_DISCLAIMER, MALECNS_PAPER, LITE_LABEL
from .config import OUT_DIR, GRAPH_NPZ
from .config import default_palette

LITE_SIZE = 4096

# region bucket codes (for slice mode + meters)
REGION_OPTIC = 0
REGION_CX = 1
REGION_SENSORY = 2
REGION_ASCENDING = 3
REGION_DESCENDING = 4
REGION_VNC = 5
REGION_MOTOR = 6
REGION_ENDOCRINE = 7
REGION_OTHER = 8

REGION_NAMES = {
    REGION_OPTIC: "optic",
    REGION_CX: "central/cx",
    REGION_SENSORY: "sensory",
    REGION_ASCENDING: "ascending",
    REGION_DESCENDING: "descending",
    REGION_VNC: "vnc",
    REGION_MOTOR: "motor",
    REGION_ENDOCRINE: "endocrine",
    REGION_OTHER: "other",
}


def _region_of(sup: str) -> int:
    if sup is None:
        return REGION_OTHER
    if sup.startswith("ol_") or sup.startswith("visual") or sup == "ENS":
        return REGION_OPTIC
    if sup.startswith("cb_sensory"):
        return REGION_SENSORY
    if sup in ("cb_motor", "vnc_motor"):
        return REGION_MOTOR
    if sup in ("cb_endocrine", "vnc_endocrine"):
        return REGION_ENDOCRINE
    if "ascending" in sup:
        return REGION_ASCENDING
    if "descending" in sup:
        return REGION_DESCENDING
    if sup.startswith("vnc_"):
        return REGION_VNC
    if sup == "cb_intrinsic":
        return REGION_CX
    if sup.startswith("cb_"):
        return REGION_CX
    if sup in ("sensory_ascending", "sensory_descending"):
        return REGION_SENSORY
    return REGION_OTHER


class Graph:
    def __init__(self, n=0, W=None, body=None, superclass=None, types=None, side=None,
                 xyz=None, pos_source=None, nt=None, fru=None, receptor=None, dim=None,
                 vnc=None, name="", is_lite=False, is_fixture=False, fallback_fraction=0.0,
                 manifest=None, superclass_names=None, type_names=None, side_names=None,
                 nt_names=None):
        self.n = n
        self.W = W
        self.body = body
        self.superclass = superclass
        self.types = types
        self.side = side
        self.xyz = xyz
        self.pos_source = pos_source
        self.nt = nt
        self.fru = fru
        self.receptor = receptor
        self.dim = dim
        self.vnc = vnc
        self.name = name
        self.is_lite = is_lite
        self.is_fixture = is_fixture
        self.fallback_fraction = fallback_fraction
        self.manifest = manifest or {}
        self.superclass_names = superclass_names or []
        self.type_names = type_names or []
        self.side_names = side_names or []
        self.nt_names = nt_names or []
        self.region_codes = None
        self.groups = {}
        self._palette = {}

    # ------------------------------------------------------------------ loaders
    @classmethod
    def load(cls, lite: bool = False, seed: int | None = None) -> "Graph":
        if not os.path.exists(GRAPH_NPZ):
            if lite:
                print("no MaleCNS cache found -> synthetic LITE BRAIN fixture")
                return synthetic_fixture()
            raise FileNotFoundError(
                "No MaleCNS cache. Run `python -m flyboard.data fetch` + "
                "`python -m flyboard.build`, or use --lite."
            )
        g = cls._from_npz(GRAPH_NPZ)
        if lite:
            g = subsample_lite(g, seed=seed or 2024)
        g.finalize()
        return g

    @classmethod
    def _from_npz(cls, path: str) -> "Graph":
        d = np.load(path, allow_pickle=False)
        n = int(d["superclass"].size)
        W = sp.csr_matrix(
            (d["W_data"], d["W_indices"], d["W_indptr"]),
            shape=(n, n),
        )
        g = cls(
            n=n,
            W=W,
            body=d["body"],
            superclass=d["superclass"],
            types=d["type"],
            side=d["side"],
            xyz=np.ascontiguousarray(d["xyz"], dtype=np.float32),
            pos_source=d["pos_source"],
            nt=d["nt_codes"],
            fru=d["fru"],
            receptor=d["receptor"],
            dim=d["dimorphism"],
            vnc=d["vnc"],
            name="MaleCNS v1.0 full net",
            fallback_fraction=float((d["pos_source"] > 0).mean()),
        )
        if "exitNerve" in d:
            g.exit_codes = d["exitNerve"]
            g.neuromere_codes = d["somaNeuromere"]
        mpath = os.path.join(OUT_DIR, "graph_manifest.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                g.manifest = json.load(f)
            g.superclass_names = g.manifest.get("superclasses", [])
            g.type_names = g.manifest.get("types", [])
            g.side_names = g.manifest.get("sides", [])
            g.nt_names = g.manifest.get("nt", [])
        return g

    def finalize(self):
        self.superclass_names = _ensure(self.superclass_names, self.superclass)
        self.type_names = _ensure(self.type_names, self.types)
        self.side_names = _ensure(self.side_names, self.side)
        self.region_codes = np.empty(self.n, dtype=np.uint8)
        for i, s in enumerate(self.superclass):
            self.region_codes[i] = _region_of(self.superclass_names[s])
        self._build_groups()
        self._palette = default_palette()
        self.colors = np.zeros((self.n, 3), dtype=np.float32)
        for s in range(len(self.superclass_names)):
            nm = self.superclass_names[s]
            c = self._palette.get(nm, (0.6, 0.6, 0.6))
            m = self.superclass == s
            self.colors[m] = c
        if self.n:
            self.xyz_min = self.xyz.min(axis=0)
            self.xyz_max = self.xyz.max(axis=0)
        self.fallback_mask = self.pos_source > 0 if self.pos_source is not None else np.zeros(self.n, dtype=bool)

    def _build_groups(self):
        g = {name: [] for name in GROUPS}
        for i in range(self.n):
            r = int(self.region_codes[i])
            g[REGION_NAMES[r]].append(i)
        self.groups = {k: np.asarray(v, dtype=np.int64) for k, v in g.items()}

    def region_name(self, code):
        return REGION_NAMES.get(int(code), "other")

    def type_name_of(self, i):
        return self.type_names[int(self.types[i])]

    def super_name_of(self, i):
        return self.superclass_names[int(self.superclass[i])]

    def banner(self):
        if self.is_fixture:
            return LITE_LABEL + " (synthetic fixture)"
        if self.is_lite:
            return LITE_LABEL
        return "MaleCNS v1.0 full net"


def _ensure(names_list, codes):
    if names_list:
        return names_list
    mx = int(np.max(codes)) + 1
    return [f"type{i}" for i in range(mx)]


GROUPS = {
    "optic": {REGION_OPTIC},
    "central/cx": {REGION_CX},
    "sensory": {REGION_SENSORY},
    "ascending": {REGION_ASCENDING},
    "descending": {REGION_DESCENDING},
    "vnc": {REGION_VNC},
    "motor": {REGION_MOTOR},
    "endocrine": {REGION_ENDOCRINE},
    "other": {REGION_OTHER},
}
GROUPS_CODE_NAME = {v: k for k, v in REGION_NAMES.items()}


def subsample_lite(g: Graph, seed: int = 2024) -> Graph:
    """Deterministic 4096-cell subsample covering all superclasses."""
    rng = np.random.default_rng(seed)
    indices = []
    sup_uniq = np.unique(g.superclass)
    chosen_pool = set()
    for s in sup_uniq:
        m = np.flatnonzero(g.superclass == s)
        k = min(len(m), max(2, LITE_SIZE // len(sup_uniq)))
        take = rng.choice(m, size=k, replace=False)
        indices.append(take)
        chosen_pool.update(int(x) for x in take)
    idx = np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)
    need = LITE_SIZE - len(idx)
    if need > 0:
        rest = np.array([i for i in range(g.n) if i not in chosen_pool], dtype=np.int64)
        rest = rng.choice(rest, size=min(need, len(rest)), replace=False)
        idx = np.concatenate([idx, rest])
    idx = np.sort(idx[:LITE_SIZE]).astype(np.int64)
    sub = g.subgraph(idx)
    sub.is_lite = True
    sub.name = LITE_LABEL
    return sub

def synthetic_fixture() -> Graph:
    """4096-cell synthetic net, only used when the MaleCNS cache is missing (--lite)."""
    rng = np.random.default_rng(7)
    n = LITE_SIZE
    sup_names = ["cb_intrinsic", "vnc_intrinsic", "ol_intrinsic", "cb_sensory",
                 "vnc_sensory", "descending_neuron", "vnc_motor", "cb_endocrine", "visual_projection"]
    nsup = len(sup_names)
    superclass = np.repeat(np.arange(nsup), np.full(nsup, (n // nsup) + (1 if n % nsup == 0 else 0)))[:n].astype(np.uint8)
    types = (np.arange(n) % 250).astype(np.int32)
    body = (np.arange(n) + 1).astype(np.uint64) * 10
    xyz = rng.standard_normal((n, 3)).astype(np.float32)
    xyz -= xyz.mean(axis=0)
    xyz[superclass == 7] *= 0.2  # endocrine cluster
    pos_source = np.zeros(n, dtype=np.uint8)
    # lognormal-ish weights
    rows = rng.integers(0, n, int(n * 40))
    cols = rng.integers(0, n, int(n * 40))
    w = rng.lognormal(0.5, 1.2, len(rows)).astype(np.float32)
    W = sp.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    g = Graph(
        n=n, W=W, body=body, superclass=superclass, types=types, side=np.zeros(n, dtype=np.uint8),
        xyz=xyz, pos_source=pos_source, nt=np.full(n, -1, dtype=np.int8),
        fru=np.zeros(n, dtype=np.uint8), receptor=np.zeros(n, dtype=np.uint8),
        dim=np.zeros(n, dtype=np.uint8), vnc=(np.arange(n) % 3 == 0),
        name=LITE_LABEL + " (synthetic fixture)", is_lite=True, is_fixture=True,
        superclass_names=sup_names,
        type_names=[f"SynT{i}" for i in range(250)],
        side_names=["?"],
        nt_names=[],
    )
    g.exit_codes = np.zeros(n, dtype=np.uint16)
    g.neuromere_codes = np.zeros(n, dtype=np.uint16)
    return g

def subgraph(self, idx) -> "Graph":
    g = Graph(
        n=len(idx),
        W=self.W[idx, :][:, idx].tocsr(),
        body=self.body[idx],
        superclass=self.superclass[idx],
        types=self.types[idx],
        side=self.side[idx],
        xyz=self.xyz[idx],
        pos_source=self.pos_source[idx],
        nt=self.nt[idx],
        fru=self.fru[idx],
        receptor=self.receptor[idx],
        dim=self.dim[idx],
        vnc=self.vnc[idx],
        name=self.name,
        is_lite=self.is_lite,
        fallback_fraction=float((self.pos_source[idx] > 0).mean()),
        manifest=self.manifest,
        superclass_names=self.superclass_names,
        type_names=self.type_names,
        side_names=self.side_names,
        nt_names=self.nt_names,
    )
    if hasattr(self, "exit_codes"):
        g.exit_codes = self.exit_codes[idx]
        g.neuromere_codes = self.neuromere_codes[idx]
    g.finalize()
    return g


Graph.subgraph = subgraph