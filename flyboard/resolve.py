"""Button -> MaleCNS type name -> cell index resolver.

After every cache load we write `data/out/hits.json`:
    button, requested_types, found_count, missing_types, proxy_used

If found_count == 0 the button still works, but only as a PROXY and the brain
banner turns orange: "PROXY: no exact types, driving superclass X".
"""
from __future__ import annotations

import fnmatch
import json
import os
import time

import numpy as np
import yaml

from .config import PRESETS_PATH, HITS_JSON

BUTTON_ORDER = ["FOOD", "PAIN", "LICK", "MATE", "ORGASM", "DEATH"]


class PresetResolve:
    def __init__(self, button, indices, report, proxy=None):
        self.button = button
        self.indices = np.asarray(indices, dtype=np.int64)
        self.report = report
        self.proxy = proxy or {}

    @property
    def found_count(self):
        return int(self.report.get("found_count", 0))

    @property
    def proxy_used(self):
        return bool(self.report.get("proxy_used", False))


def apply_selector(g, sel: dict):
    """Return boolean mask over graph cells for one selector dict (ANDed keys)."""
    n = g.n
    if len(sel) == 0:
        return np.ones(n, dtype=bool)
    mask = np.ones(n, dtype=bool)
    tn = np.array(g.type_names, dtype=object)
    sn = np.array(g.superclass_names, dtype=object)
    for key, val in sel.items():
        if val is None:
            continue
        vals = [val] if isinstance(val, str) else list(val)
        if key == "type_glob":
            m = np.zeros(n, dtype=bool)
            for tg in vals:
                for ix, t in enumerate(g.types):
                    if fnmatch.fnmatch(tn[t], tg):
                        m[ix] = True
            mask &= m
        elif key == "super":
            m = np.zeros(n, dtype=bool)
            for s in vals:
                m |= sn[g.superclass] == s
            mask &= m
        elif key == "receptor":
            m = np.zeros(n, dtype=bool)
            for r in vals:
                m |= g.receptor == r
            mask &= m
        elif key == "fru":
            m = np.zeros(n, dtype=bool)
            for f in vals:
                m |= g.fru == {"fru_high": 1, "fru_low": 2, "coexpress_high": 3,
                               "coexpress_low": 4, "dsx_high": 5, "dsx_low": 6}.get(f, 99)
            mask &= m
        elif key == "dimorphism":
            m = np.zeros(n, dtype=bool)
            for d in vals:
                code = {"male-specific": 1, "potentially male-specific": 2,
                        "sexually dimorphic": 3, "potentially sexually dimorphic": 4}.get(d, 99)
                m |= g.dim == code
            mask &= m
        elif key == "nt":
            m = np.zeros(n, dtype=bool)
            for tgt in vals:
                m |= g.nt == tgt
            mask &= m
        elif key == "nt_glob":
            m = np.zeros(n, dtype=bool)
            nt = np.array(g.nt_names, dtype=object)
            for tg in vals:
                for ix in range(g.n):
                    if g.nt[ix] >= 0 and g.nt[ix] < len(nt) and fnmatch.fnmatch(str(nt[g.nt[ix]]), tg):
                        m[ix] = True
            mask &= m
        elif key == "exit_nerve":
            m = np.zeros(n, dtype=bool)
            ext = np.array(g.manifest.get("exitNerves", [""]), dtype=object)
            codes = getattr(g, "exit_codes", np.zeros(n, dtype=np.uint16))
            for en in vals:
                if en in ext:
                    m |= codes == ext.tolist().index(en)
            mask &= m
        elif key == "neuromere":
            m = np.zeros(n, dtype=bool)
            exn = np.array(g.manifest.get("neuromeres", [""]), dtype=object)
            codes = getattr(g, "neuromere_codes", np.zeros(n, dtype=np.uint16))
            for nm in vals:
                if nm in exn:
                    m |= codes == exn.tolist().index(nm)
            mask &= m
        else:
            raise KeyError(f"unknown selector {key!r}")
    return mask


def tier_match(g, tier):
    """A tier is an OR of selectors; each selector dict is ANDed internally."""
    m = np.zeros(g.n, dtype=bool)
    labels = []
    for sel_ in tier:
        if not isinstance(sel_, dict) or not sel_:
            continue
        m |= apply_selector(g, sel_)
        for k, v in sel_.items():
            labels.append(f"{k}={v}")
    return m, labels


def resolve_button(g, button_cfg: dict, button_name: str) -> PresetResolve:
    requested = []
    for tier in button_cfg.get("tiers", []):
        for sel in tier:
            for k, v in sel.items():
                label = f"{k}={v}" if not isinstance(v, str) else f"{k}={v}"
                requested.append(label)
    for tier in button_cfg.get("tiers", []):
        m, _ = tier_match(g, tier)
        cnt = int(m.sum())
        if cnt > 0:
            return resolve_done(g, button_name, requested, np.flatnonzero(m), tier, proxy=False, cfg=button_cfg)
    return proxy_resolve(g, button_name, requested, button_cfg)


def proxy_resolve(g, button_name, requested, cfg):
    proxy = cfg.get("proxy", {})
    m = apply_selector(g, proxy)
    take = int(proxy.get("take", 120))
    idx = np.flatnonzero(m)
    if len(idx) > take:
        rng = np.random.default_rng(len(button_name) * 7 + 3)
        idx = rng.choice(idx, size=take, replace=False)
    return resolve_done(g, button_name, requested, idx, None, proxy=True, cfg=cfg, used_supers=proxy.get("super", []))


def resolve_done(g, button_name, requested, indices, tier, proxy, cfg, used_supers=None):
    found_names = []
    for i in indices:
        t = g.type_name_of(i)
        if t not in found_names:
            found_names.append(t)
    missing = []
    if tier is not None:
        for sel in tier:
            for k, v in sel.items():
                m = apply_selector(g, {k: v} if isinstance(v, str) else {k: v})
                if int(m.sum()) == 0 and v is not None:
                    missing.append(f"{k}={v}")
    report = {
        "button": button_name,
        "requested_types": requested,
        "found_count": int(len(indices)),
        "missing_types": missing,
        "proxy_used": proxy,
        "proxy_superclasses": used_supers or (list(cfg.get("proxy", {}).get("super", [])) if proxy else []),
        "hold_ms": float(cfg.get("hold_ms", 250.0)),
        "extra_noise": bool(cfg.get("extra_noise", False)),
        "bounded_extra_drive": bool(cfg.get("bounded_extra_drive", False)),
        "found_sample_types": found_names[:12],
        "n_types": len(found_names),
    }
    return PresetResolve(button_name, indices, report, proxy=cfg.get("proxy", {}))


def load_presets(path=None):
    path = path or PRESETS_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_all(g, path=None) -> dict:
    presets = load_presets(path)
    out = {}
    for name in BUTTON_ORDER:
        if name == "DEATH":
            out[name] = None
            continue
        cfg = presets["buttons"][name]
        res = resolve_button(g, cfg, name)
        out[name] = res
    return out


def write_hits(g, resolved: dict | None = None, path=None) -> dict:
    if resolved is None:
        resolved = resolve_all(g)
    path = path or HITS_JSON
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "graph": g.banner(),
        "cells": int(g.n),
        "buttons": {},
    }
    for name, r in resolved.items():
        if r is None:
            continue
        rep = dict(r.report)
        doc["buttons"][name] = rep
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return doc


def run_resolve(g):
    resolved = resolve_all(g)
    write_hits(g, resolved)
    for name, r in resolved.items():
        if r is None:
            continue
        flag = "PROXY" if r.proxy_used else "ok   "
        print(f"  [{flag}] {name:8s} {r.found_count:5d} cells  {r.report['found_sample_types'][:5]}")
    return resolved