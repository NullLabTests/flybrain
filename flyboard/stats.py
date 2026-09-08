"""'Who lit up' panel computation.

Light-up score = (spikes during hold + 300 ms) - baseline (rolling 1 s), computed
per cell; idle noise does not count as a reward flood.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import ORGASM_ACTIVE_FRACTION_CAP

# A cell "lit up" if its burst extra beat the rolling baseline by >=1 spike.
LIT_THRESHOLD = 1.0


@dataclass
class PanelData:
    top_types: list = field(default_factory=list)      # (name, extra)
    top_super: list = field(default_factory=list)      # (name, extra)
    sparkline: np.ndarray = field(default_factory=lambda: np.zeros(256))
    seed_types: list = field(default_factory=list)     # (name, count) injected
    downstream_types: list = field(default_factory=list)  # (name, extra) not injected
    n_downstream_cells: int = 0
    total_extra: float = 0.0
    burst_total_spikes: list = field(default_factory=list)
    active_fraction_cap: float = 0.0
    active_fraction: float = 0.0
    measure_active: bool = False
    button: str = ""


def compute_panel(graph, sim, top_types=15, top_super=8) -> PanelData:
    extra = sim.leaderboard()
    sim.lead_extra = extra
    g = graph
    t = g.types
    s = g.superclass

    use = extra.copy()
    # consider only cells that actually lit up (beat baseline by >=1 spike)
    use[use < LIT_THRESHOLD] = 0.0
    order = np.argsort(-use, kind="stable")

    agg = {}
    for i in order:
        if use[i] <= 0:
            break
        tt = int(t[i])
        agg[tt] = agg.get(tt, 0.0) + float(use[i])
    top_t = sorted(agg.items(), key=lambda kv: -kv[1])[:top_types]
    top_types_list = [(g.type_names[int(k)] or "untyped", float(v)) for k, v in top_t]

    sagg = {}
    for i in order:
        if use[i] <= 0:
            break
        ss = int(s[i])
        sagg[ss] = sagg.get(ss, 0.0) + float(use[i])
    top_s = sorted(sagg.items(), key=lambda kv: -kv[1])[:top_super]
    top_super_list = [(g.superclass_names[int(k)] or "untyped", float(v)) for k, v in top_s]

    seed = set(int(i) for i in sim.inject_indices)
    ds_idx = np.flatnonzero(use > 0)
    ds_cells = [int(i) for i in ds_idx if i not in seed]
    tns = np.array(g.type_names, dtype=object)
    seed_cnt = {}
    for i in seed:
        if use[i] <= 0:
            continue
        seed_cnt[str(tns[int(t[i])] or "untyped")] = seed_cnt.get(str(tns[int(t[i])] or "untyped"), 0) + 1
    ds_agg = {}
    for i in ds_cells:
        tt = str(tns[int(t[i])] or "untyped")
        ds_agg[tt] = ds_agg.get(tt, 0.0) + float(use[i])
    ds_top = sorted(ds_agg.items(), key=lambda kv: -kv[1])[:top_types]

    spark = sim.spark
    idx = sim.spark_pos
    ordered = np.concatenate([spark[idx:], spark[:idx]])

    return PanelData(
        top_types=top_types_list,
        top_super=top_super_list,
        sparkline=ordered,
        seed_types=sorted(seed_cnt.items(), key=lambda kv: -kv[1])[:top_types],
        downstream_types=ds_top,
        n_downstream_cells=len(ds_cells),
        total_extra=float(use.sum()),
        burst_total_spikes=sim.burst_spikes.tolist() if sim.burst_spikes.size else [],
        active_fraction=sim.active_fraction(),
        active_fraction_cap=ORGASM_ACTIVE_FRACTION_CAP,
        measure_active=sim.measure_active,
        button=sim.last_button or "",
    )