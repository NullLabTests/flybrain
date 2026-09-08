"""Download the real MaleCNS v1.0 cache files (Janelia public GCS bucket).

Usage:
    python -m flyboard.data fetch            # download all missing cache files
    python -m flyboard.data fetch --force
    python -m flyboard.data status
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

from .config import (
    CACHE_DIR,
    ANN_FEATHER,
    NTR_FEATHER,
    W_FEATHER,
    SOMA_DIR,
    SOMA_INFO,
    SOMA_SHARDS,
    SOMA_SHARD_SIZES_BYTES,
    ANNOTATIONS_REMOTE,
    NEUROTRANS_REMOTE,
    WEIGHTS_REMOTE,
    SOMA_PREFIX,
    CACHE_FILES,
)


def _fetch(url: str, dest: str, expected: int | None, force: bool) -> bool:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if (not force) and os.path.exists(dest):
        size = os.path.getsize(dest)
        if expected is None or size == expected:
            print(f"  ok   (cached)  {os.path.basename(dest)}")
            return True
        print(f"  bad  (size {size} != {expected}) redownload {os.path.basename(dest)}")
    print(f"  get          {url}")
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "flyboard/1.0"})
    with urllib.request.urlopen(req, timeout=240) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    got = os.path.getsize(dest)
    if expected is not None and got != expected:
        os.remove(dest)
        raise RuntimeError(f"download incomplete for {dest}: {got} != {expected}")
    print(f"  done         {dest} ({got} bytes)")
    return True


def ensure_soma(force: bool = False) -> None:
    base = os.path.join(CACHE_DIR, SOMA_DIR)
    os.makedirs(os.path.join(base, "by_id"), exist_ok=True)
    os.makedirs(os.path.join(base, "by_rel_body"), exist_ok=True)
    os.makedirs(os.path.join(base, "by_rel_nucleus_id"), exist_ok=True)
    os.makedirs(os.path.join(base, "by_spatial_level_0"), exist_ok=True)
    _fetch(SOMA_PREFIX + "/info", os.path.join(base, SOMA_INFO), None, force)
    for key, rel in SOMA_SHARDS.items():
        _fetch(
            SOMA_PREFIX + f"/{key}/0.shard",
            os.path.join(base, rel),
            SOMA_SHARD_SIZES_BYTES[key],
            force,
        )


def run_fetch(force: bool = False) -> None:
    print("== MaleCNS v1.0 cache ==")
    files = {
        ANN_FEATHER: (ANNOTATIONS_REMOTE, CACHE_FILES[ANN_FEATHER]),
        NTR_FEATHER: (NEUROTRANS_REMOTE, CACHE_FILES[NTR_FEATHER]),
        W_FEATHER: (WEIGHTS_REMOTE, CACHE_FILES[W_FEATHER]),
    }
    for name, (url, size) in files.items():
        _fetch(url, os.path.join(CACHE_DIR, name), size, force)
    print("== soma points ==")
    ensure_soma(force)
    print("cache ready at", CACHE_DIR)


def status() -> int:
    missing = []
    total = 0
    items = [
        (ANN_FEATHER, CACHE_FILES[ANN_FEATHER]),
        (NTR_FEATHER, CACHE_FILES[NTR_FEATHER]),
        (W_FEATHER, CACHE_FILES[W_FEATHER]),
        (SOMA_INFO, None),
    ] + [(rel, SOMA_SHARD_SIZES_BYTES[k]) for k, rel in SOMA_SHARDS.items()]
    for rel, size in items:
        p = os.path.join(CACHE_DIR, rel)
        exists = os.path.exists(p)
        ok = exists and (size is None or os.path.getsize(p) == size)
        total += int(ok)
        if not ok:
            missing.append(rel)
        print(f"  {'OK  ' if ok else 'MISS'} {rel}")
    print(f"cache: {total}/{len(items)} files present")
    if missing:
        print("run:  python -m flyboard.data fetch")
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="flyboard.data", description="FLYBOARD data fetch")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("fetch").add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "status":
        return status()
    run_fetch(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())