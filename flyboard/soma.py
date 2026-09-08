"""Extract measured soma positions from the MaleCNS v1.0 soma-points neuroglancer
annotation shards (EM 8nm coordinates).

Relies on the format readers shipped with the `neuroglancer` + `tensorstore`
packages; no hand-rolled binary parsing. Output is written to
`data/out/soma.npz` with one row per soma point:
    body        uint64 body segment id
    x,y,z       float32 EM voxel coordinates (8nm units)
    kind        0 = nucleus soma, 1 = tosoma
"""
from __future__ import annotations

import os
import sys

import numpy as np

from .config import CACHE_DIR, OUT_DIR, SOMA_DIR


def collect_soma_points(soma_root: str):
    from neuroglancer.read_precomputed_annotations import AnnotationReader

    reader = AnnotationReader(f"file://{soma_root}/")
    anns = list(reader.get_within_spatial_bounds())
    if not anns:
        raise RuntimeError("no soma annotations found in " + soma_root)

    bodies = []
    coords = []
    kinds = []
    import tensorstore as ts

    batch = ts.Batch()
    fut = [reader.by_id.get(int(a.id), batch=batch) for a in anns]
    for a, f in zip(anns, fut):
        v = f.result()
        if v is None or not v.segments or not v.segments[0]:
            continue
        bodies.append(int(v.segments[0][0]))
        coords.append([float(c) for c in a.point[:3]])
        kinds.append(int(a.props[0]) if len(a.props) else 0)
    n = len(bodies)
    return n, np.array(bodies, dtype=np.uint64), np.array(coords, dtype=np.float32), np.array(kinds, dtype=np.int8)


def build_soma_cache(force: bool = False) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "soma.npz")
    if os.path.exists(out) and not force:
        print("soma cache up to date:", out)
        return out
    soma_root = os.path.join(CACHE_DIR, SOMA_DIR)
    if not os.path.exists(os.path.join(soma_root, "info")):
        raise RuntimeError(
            "soma shards not downloaded; run `python -m flyboard.data fetch`"
        )
    n, body, xyz, kind = collect_soma_points(soma_root)
    print(f"soma cache: {n} points")
    np.savez_compressed(out, body=body, xyz=xyz, kind=kind)
    return out


if __name__ == "__main__":
    build_soma_cache(force="--force" in sys.argv)