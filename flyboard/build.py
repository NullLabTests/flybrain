"""Build the FLYBOARD graph cache from real MaleCNS v1.0 tables.

Produces `data/out/flyboard_graph.npz` (+ `graph_manifest.json`):
  - cell table for the 166,700 neurons with a measured superclass
  - directed connectome weights (synapse counts from the MaleCNS flat-connectome
    table), column-normalized per presynaptic cell so the toy integrator is stable
  - measured soma positions (MaleCNS soma-points shards); cells without a soma get
    a FALLBACK POS (sibling-type mean, else seeded procedural) and are flagged.

Run:  python -m flyboard.build [--force]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import scipy.sparse as sp

from . import FOOTER_DISCLAIMER, MALECNS_PAPER, LITE_LABEL, SIM_W_GAIN
from .config import (
    OUT_DIR,
    CACHE_DIR,
    ANN_FEATHER,
    NTR_FEATHER,
    W_FEATHER,
)


def _ipc(path: str):
    src = pa.memory_map(path)
    reader = pa.ipc.open_file(src)
    return src, reader


def load_annotations():
    src, reader = _ipc(os.path.join(CACHE_DIR, ANN_FEATHER))
    nb = reader.num_record_batches
    parts = {c: [] for c in ("body", "superclass", "type", "flywireType", "side",
                             "dimorphism", "fruDsx", "receptorType", "exitNerve",
                             "somaNeuromere", "statusLabel")}
    for b in range(nb):
        batch = reader.get_batch(b)
        cols = batch.schema.names
        for nm in parts:
            if nm == "body":
                src_col = "bodyId" if "bodyId" in cols else "body"
            elif nm == "side":
                src_col = "somaSide" if "somaSide" in cols else None
            else:
                src_col = nm if nm in cols else None
            if src_col is None:
                continue
            col = batch.column(src_col)
            if pa.types.is_integer(col.type):
                parts[nm].append(col.to_numpy(zero_copy_only=False))
            else:
                parts[nm].append(np.asarray(col.to_pylist(), dtype=object))
    src.close()

    body = np.concatenate([p.astype(np.int64) for p in parts["body"]])
    sup_raw = np.concatenate([p for p in parts["superclass"]])

    def col(name):
        vals = [x if isinstance(x, str) else "" for p in parts[name] for x in p]
        return np.asarray(vals, dtype=object)

    nrow = len(body)
    mask = np.array([s is not None and isinstance(s, str) for s in sup_raw], dtype=bool)
    sup = np.asarray(sup_raw, dtype=object)[mask].astype(str)
    cells = dict(
        body=body[mask],
        superclass=sup,
        type=col("type")[mask].astype(str),
        flywire_type=col("flywireType")[mask].astype(str),
        side=col("side")[mask].astype(str),
        dimorphism=col("dimorphism")[mask].astype(str),
        fruDsx=col("fruDsx")[mask].astype(str),
        receptorType=col("receptorType")[mask].astype(str),
        exitNerve=col("exitNerve")[mask].astype(str),
        somaNeuromere=col("somaNeuromere")[mask].astype(str),
        status=col("statusLabel")[mask].astype(str),
    )
    n = len(cells["body"])
    print(f"annotations: {nrow} bodies -> {n} neurons with superclass")
    return cells, n


def load_neurotransmitters_head():
    p = os.path.join(CACHE_DIR, NTR_FEATHER)
    if not os.path.exists(p):
        print("neurotransmitters table missing; skipping NT coloring")
        return None, []
    src, reader = _ipc(p)
    body_parts = []
    pred_parts = []
    conf_parts = []
    for b in range(reader.num_record_batches):
        batch = reader.get_batch(b)
        body_parts.append(batch.column("body").to_numpy(zero_copy_only=False).astype(np.int64))
        pred_parts.append(batch.column("predicted_nt").to_pylist())
        conf_parts.append(batch.column("predicted_nt_confidence").to_numpy(zero_copy_only=False))
    src.close()
    body = np.concatenate(body_parts)
    pred = []
    for p in pred_parts:
        pred.extend(x if isinstance(x, str) else "" for x in p)
    conf = np.concatenate(conf_parts)
    return (body, np.asarray(pred, dtype=object), conf), sorted(set(pred))


def choose_transmitter(nt_head, cell_body):
    if nt_head is None:
        return np.full(len(cell_body), -1, dtype=np.int8), []
    body, pred, conf = nt_head
    names = sorted({x for x in pred})
    name_to_code = {nm: i for i, nm in enumerate(names)}
    idm = np.argsort(body)
    srt = body[idm]
    idx = np.clip(np.searchsorted(srt, cell_body), 0, len(idm) - 1)
    mapped = idm[idx]
    ok = srt[idx] == cell_body
    out = np.full(len(cell_body), -1, dtype=np.int8)
    confs = conf[mapped]
    for i in range(len(cell_body)):
        if ok[i] and confs[i] >= 0.35 and pred[mapped[i]] in name_to_code:
            out[i] = name_to_code[pred[mapped[i]]]
    return out, names


def load_weights(cells_body):
    src, reader = _ipc(os.path.join(CACHE_DIR, W_FEATHER))
    schema = reader.schema
    names = schema.names
    print("weights columns:", names)
    pcol = "body_pre" if "body_pre" in names else ("pre_body" if "pre_body" in names else names[0])
    qcol = "body_post" if "body_post" in names else ("post_body" if "post_body" in names else names[1])
    wcol = "weight" if "weight" in names else names[-1]

    srt = np.argsort(cells_body)
    sorted_ids = cells_body[srt]

    def map_ids(ids):
        i = np.searchsorted(sorted_ids, ids)
        i = np.clip(i, 0, len(sorted_ids) - 1)
        hit = sorted_ids[i] == ids
        return srt[i], hit

    pre_all, post_all, w_all = [], [], []
    nb = reader.num_record_batches
    for b in range(nb):
        batch = reader.get_batch(b)
        pre = batch.column(pcol).to_numpy(zero_copy_only=False).astype(np.int64)
        post = batch.column(qcol).to_numpy(zero_copy_only=False).astype(np.int64)
        if wcol in batch.schema.names:
            w = batch.column(wcol).to_numpy(zero_copy_only=False).astype(np.float64)
        else:
            w = np.ones(len(pre), dtype=np.float64)
        ip, ip_hit = map_ids(pre)
        iv, iv_hit = map_ids(post)
        m = ip_hit & iv_hit & (w > 0.0)
        pre_all.append(ip[m].astype(np.int32))
        post_all.append(iv[m].astype(np.int32))
        w_all.append(w[m].astype(np.float32))
        if b % max(1, nb // 10) == 0:
            print(f"  weights batch {b + 1}/{nb}")
    src.close()
    pre = np.concatenate(pre_all)
    post = np.concatenate(post_all)
    w = np.concatenate(w_all)
    del pre_all, post_all, w_all
    return pre, post, w


def load_soma_positions(cells_body):
    p = os.path.join(OUT_DIR, "soma.npz")
    if not os.path.exists(p):
        return None, None
    d = np.load(p)
    sbody = d["body"].astype(np.int64)
    sxyz = d["xyz"]
    srt = np.sort(sbody)
    idx = np.clip(np.searchsorted(srt, cells_body), 0, len(srt) - 1)
    ok = srt[idx] == cells_body
    pos = np.full((len(cells_body), 3), np.nan, np.float32)
    ids = np.where(ok)[0]
    pos[ids] = sxyz[idx[ids]]
    return pos, ok


def fill_fallback_positions(cells_type, pos, ok):
    n = len(pos)
    src = np.zeros(n, dtype=np.uint8)  # 0 = measured soma
    need = ~ok
    types = np.array(cells_type, dtype=object)
    utypes, inv = np.unique(types, return_inverse=True)
    sums = np.zeros((len(utypes), 3), dtype=np.float64)
    cnt = np.zeros(len(utypes), dtype=np.int64)
    np.add.at(sums, inv[ok], pos[ok])
    np.add.at(cnt, inv[ok], 1)
    with_mean = cnt > 0
    means = np.full((len(utypes), 3), np.nan, dtype=np.float64)
    means[with_mean] = sums[with_mean] / cnt[with_mean][:, None]
    assign = need & with_mean[inv]
    pos[assign] = means[inv[assign]]
    src[assign] = 1
    need[assign] = False
    rem = np.where(need)[0]
    if len(rem):
        rng = np.random.default_rng(1337)
        lo = np.nanmin(pos, axis=0)
        hi = np.nanmax(pos, axis=0)
        span = hi - lo
        u = rng.standard_normal((len(rem), 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        r = span.max() * (0.35 + 0.15 * rng.random(len(rem)))[:, None]
        pos[rem] = r * u + (lo + hi) / 2
        src[rem] = 2
    return pos, src


def build(force: bool = False):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_npz = os.path.join(OUT_DIR, "flyboard_graph.npz")
    if os.path.exists(out_npz) and not force:
        print("graph cache up to date:", out_npz)
        return out_npz
    t0 = time.time()
    cells, n = load_annotations()
    body = cells["body"].astype(np.int64)

    pre, post, w = load_weights(body)
    W = sp.coo_matrix((w, (pre, post)), shape=(n, n)).tocsr()
    del pre, post, w
    colsum = np.asarray(W.sum(axis=0)).ravel()
    inv = np.zeros_like(colsum)
    nz = colsum > 0
    inv[nz] = 1.0 / colsum[nz]
    W.data = W.data * inv[W.indices]
    print(f"graph: {W.nnz} edges, {n} cells, column-normalized per postsynaptic cell")

    pos, ok = load_soma_positions(body)
    if pos is None:
        pos = np.full((n, 3), np.nan, np.float32)
        ok = np.zeros(n, dtype=bool)
        print("no soma cache; all positions will be FALLBACK POS")
    pos, src = fill_fallback_positions(cells["type"], pos, ok)
    n_fallback = int((src > 0).sum())
    print(f"positions: {n - n_fallback} measured soma, {n_fallback} fallback")

    nt_head, nt_names = load_neurotransmitters_head()
    nt_codes, _ = choose_transmitter(nt_head, body)

    sup = cells["superclass"]
    sup_names = sorted(set(sup))
    sup_code = np.array([sup_names.index(s) for s in sup], dtype=np.uint8)
    typ_names = sorted(set(cells["type"]))
    type_dict = {x: i for i, x in enumerate(typ_names)}
    type_code = np.array([type_dict[s] for s in cells["type"]], dtype=np.int32)
    side_names = sorted(set(cells["side"]))
    side_code = np.array([side_names.index(s) for s in cells["side"]], dtype=np.uint8)

    fru = np.zeros(n, dtype=np.uint8)
    fru[cells["fruDsx"] == "fru_high"] = 1
    fru[cells["fruDsx"] == "fru_low"] = 2
    fru[cells["fruDsx"] == "coexpress_high"] = 3
    fru[cells["fruDsx"] == "coexpress_low"] = 4
    fru[cells["fruDsx"] == "dsx_high"] = 5
    fru[cells["fruDsx"] == "dsx_low"] = 6

    exit_names = sorted({x for x in cells["exitNerve"] if x})
    exit_names = ["", ] + exit_names
    exit_code = np.array([exit_names.index(x) if x in exit_names else 0 for x in cells["exitNerve"]], dtype=np.uint16)
    nmr_names = sorted({x for x in cells["somaNeuromere"] if x})
    nmr_names = ["", ] + nmr_names
    nmr_code = np.array([nmr_names.index(x) if x in nmr_names else 0 for x in cells["somaNeuromere"]], dtype=np.uint16)

    rec = np.zeros(n, dtype=np.uint8)
    for i, v in enumerate(cells["receptorType"]):
        if "ppk25" in v:
            rec[i] = 1
        elif "ppk23" in v:
            rec[i] = 2
        elif "IR52b" in v:
            rec[i] = 3

    dim = np.zeros(n, dtype=np.uint8)
    dim[cells["dimorphism"] == "male-specific"] = 1
    dim[cells["dimorphism"] == "potentially male-specific"] = 2
    dim[cells["dimorphism"] == "sexually dimorphic"] = 3
    dim[cells["dimorphism"] == "potentially sexually dimorphic"] = 4

    vnc = np.array([s.startswith("vnc_") for s in sup], dtype=bool)

    np.savez_compressed(
        out_npz,
        body=body.astype(np.uint64),
        superclass=sup_code,
        type=type_code,
        side=side_code,
        dimorphism=dim,
        fru=fru,
        receptor=rec,
        vnc=vnc,
        xyz=pos.astype(np.float32),
        pos_source=src.astype(np.uint8),
        exitNerve=exit_code,
        somaNeuromere=nmr_code,
        W_data=W.data.astype(np.float32),
        W_indices=W.indices.astype(np.int32),
        W_indptr=W.indptr.astype(np.int32),
        nt_codes=nt_codes,
    )
    del W

    manifest = {
        "cells": int(n),
        "edges": int(os.path.getsize(out_npz)),  # placeholder, overwritten
        "org": "MaleCNS v1.0 (Janelia FlyEM + Cambridge + Google Research)",
        "population": "neurons with measured superclass",
        "weights": "directed synapse counts, column-normalized per postsynaptic cell",
        "nt": nt_names,
        "superclasses": sup_names,
        "types": typ_names,
        "sides": side_names,
        "exitNerves": exit_names,
        "neuromeres": nmr_names,
        "n_fallback_pos": n_fallback,
        "banner_full": "MaleCNS v1.0 full net",
        "banner_lite": LITE_LABEL,
        "footer": FOOTER_DISCLAIMER,
        "paper": MALECNS_PAPER,
        "dt_s": 0.02,
        "tau_s": 0.1,
        "scale": SIM_W_GAIN,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sha256": hashlib.sha256(open(out_npz, "rb").read()).hexdigest()[:16],
        "elapsed_s": round(time.time() - t0, 1),
    }
    manifest["edges"] = int(np.load(out_npz)["W_data"].size)
    with open(os.path.join(OUT_DIR, "graph_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("built", out_npz)
    print(json.dumps(manifest, indent=2))
    return out_npz


def main(argv=None):
    force = "--force" in (argv or sys.argv[1:])
    build(force=force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())