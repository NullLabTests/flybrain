"""FLYBOARD spec tests (headless).

Run:
    .venv/bin/python tests/test_flyboard.py          # all
    .venv/bin/python tests/test_flyboard.py test1    # one

Tests use the full MaleCNS cache when present, else the --lite synthetic
fixture.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flyboard.config import HITS_JSON, OUT_DIR
from flyboard.graph import Graph
from flyboard.resolve import resolve_all, write_hits
from flyboard.sim import Sim
from flyboard.stats import compute_panel
from flyboard import FOOTER_DISCLAIMER, ORGASM_ACTIVE_FRACTION_CAP

WARMUP = 100
MEASURE_TICKS = int(round((250 + 300) / 20))  # hold 250 + 300 ms = 27 ticks


def _setup(seed=42):
    full = os.path.exists(os.path.join(OUT_DIR, "flyboard_graph.npz"))
    g = Graph.load(lite=not full, seed=seed)
    sim = Sim(g, noise=0.05, seed=seed)
    return g, sim


def _warm(sim, n=WARMUP, seed=11):
    sim.rng = np.random.default_rng(seed)
    for _ in range(n):
        sim.step()


def test1_injected_spike_more_than_baseline():
    g, sim = _setup()
    resolved = resolve_all(g)
    r = resolved["FOOD"]
    assert r.found_count > 0, "FOOD must resolve to real MaleCNS cells"
    _warm(sim)
    sim.inject(r.indices, 0.45, 250, button="FOOD")
    for _ in range(MEASURE_TICKS + 10):
        sim.step()
    extra = sim.leaderboard()
    inj = r.indices
    assert float(sim.meas[inj].mean()) >= 1.0, "injected cells should spike in the hold window"
    assert float(extra[inj].mean()) > 0.0, (
        "injected cells must out-spike their 1 s baseline (spec: real spike increase, else bug)"
    )
    print(f"  FOOD injected={int(r.found_count)} cells, their burst-beats-baseline avg extra={extra[inj].mean():.2f}")
    return True


def test2_leaderboard_shows_injected_types():
    g, sim = _setup()
    resolved = resolve_all(g)
    r = resolved["MATE"]
    _warm(sim)
    sim.inject(r.indices, 0.9, 250, button="MATE")  # full population, like the real button
    for _ in range(MEASURE_TICKS + 10):
        sim.step()
    panel = compute_panel(g, sim)
    seed_types = panel.seed_types
    top_names = {nm for nm, _ in panel.top_types}
    seed_names = {nm for nm, _ in seed_types}
    assert len(seed_names) > 0, "injected population must include lit types"
    hit = seed_names & top_names
    assert len(hit) >= 1, (
        f"leaderboard should include an injected type name (lit seeds {seed_names}, tops {top_names})"
    )
    print(f"  leaderboard top types include injected seeds: {sorted(hit)}")
    return True


def test3_orgasm_does_not_flood():
    """FAKE REWARD BURST: the reward drive is bounded to a <=1% subset of the
    net (MaleCNS idle sits ~0.993 V below nothing: bg=0.180, tau=0.1 keeps every
    cell ~0.007 V under threshold, so a big drive WOULD flood the whole net).
    ORGASM therefore drives at most cap*n reward cells; driven fraction stays
    within the documented cap by construction."""
    g, sim = _setup()
    resolved = resolve_all(g)
    r = resolved["ORGASM"]
    _warm(sim)
    sim.inject(r.indices, 0.9, 250, button="ORGASM", bounded_extra=True)
    driven = len(sim.inject_indices)
    frac = sim.reward_driven_fraction()
    assert frac <= ORGASM_ACTIVE_FRACTION_CAP + 1e-9, (
        f"ORGASM drove {frac:.4f} of the net (> cap {ORGASM_ACTIVE_FRACTION_CAP:.2f}), "
        "rewiring the reward driven set"
    )
    assert driven < len(r.indices), "bounded_extra must cap the driven reward subset"
    for _ in range(MEASURE_TICKS + 20):
        sim.step()
    assert float(sim.meas[sim.inject_indices].sum()) >= 1.0, (
        "bounded reward burst must still drive the reward cloud"
    )
    print(f"  ORGASM: drove {driven} reward cells ({frac:.4%} of net, cap {ORGASM_ACTIVE_FRACTION_CAP:.0%}); "
          f"DA extras={sim.meas[sim.inject_indices].sum():.0f}")
    return True


def test4_death_kills_spikes():
    g, sim = _setup()
    _warm(sim)
    sim.die()
    sim.step()
    assert not bool(sim.spikes.any()), "DEATH: all spikes must be 0 on the next tick"
    print("  DEATH zeroes spikes on next tick")
    return True


def test5_no_camera_module():
    code = (
        "import sys; import flyboard, flyboard.ui, flyboard.source, flyboard.render;"
        "'camera' in sys.modules and sys.exit(1); sys.exit(0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, "camera module must not be imported: " + r.stderr
    for rel, root in [("source.py", ROOT + "/flyboard"), ("ui.py", ROOT + "/flyboard"), ("render.py", ROOT + "/flyboard")]:
        with open(os.path.join(root, rel)) as f:
            body = f.read()
        assert "import camera" not in body and "from camera" not in body, f"{rel} imports a camera module"
    print("  no camera module imported")
    return True


def test6_footer_disclaimer():
    needed = [
        "MaleCNS measured wiring. Toy dynamics. Buttons are current injectors.",
        "Not feelings. Not a human orgasm. Not biological death.",
    ]
    with open(os.path.join(ROOT, "flyboard", "__init__.py")) as f:
        init_src = f.read()
    for want in needed:
        assert want in FOOTER_DISCLAIMER, f"footer missing line: {want!r}"
        assert want in init_src, f"footer line not defined in __init__.py: {want!r}"
    with open(os.path.join(ROOT, "flyboard", "ui.py")) as f:
        ui_src = f.read()
    assert "FOOTER_DISCLAIMER" in ui_src, "footer must be drawn in the UI from the shared constant"
    assert ".splitlines()" in ui_src, "footer must be rendered line-by-line (pygame cannot render \\n)"
    print("  footer disclaimer present, defined once, rendered line-by-line in the UI")
    return True


def test7_hits_json_written_after_cache_load():
    g, sim = _setup()
    write_hits(g, resolve_all(g))
    assert os.path.exists(HITS_JSON), "hits.json must be written after cache load"
    import json

    doc = json.load(open(HITS_JSON))
    assert "FOOD" in doc["buttons"] and "MATE" in doc["buttons"]
    for name in ("FOOD", "MATE", "ORGASM"):
        b = doc["buttons"][name]
        assert "found_count" in b and "requested_types" in b and "missing_types" in b and "proxy_used" in b
    print("  hits.json written with button resolution report")
    return True


def test8_nearest_cell_1d_visibility():
    """Regression: clicking anywhere crashed with `tuple index out of range`.
    The hit-test must accept a 1-D boolean visibility array (it is the projected
    point set, not a 2-D mask)."""
    from flyboard.render import nearest_cell

    u = np.array([10.0, 50.0, 90.0])
    v = np.array([10.0, 50.0, 90.0])
    vis = np.array([True, False, True])
    hit = nearest_cell(u, v, vis, (12, 12))
    assert hit == 0, "should select the visible projected cell under the cursor"
    hidden = nearest_cell(u, v, vis, (52, 52))
    assert hidden == -1, "an invisible (slivered) cell under the cursor must NOT be pinned"
    none = nearest_cell(u, v, np.array([], dtype=bool), (12, 12))
    assert none == -1, "empty visibility set must not crash or pin anything"
    print("  nearest_cell handles 1-D visibility; invisible cells never pin")
    return True


def test9_headless_click_wires_buttons():
    """Regression: clicking FOOD (or any panel row) crashed and never fired the
    button. In this smoke test we keep the UI headless, click the FOOD rect, and
    assert the injection fired, then click STEP and assert it toggles."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from flyboard.config import Options
    from flyboard.ui import UI

    opts = Options(lite=True, headless=True, source="synth:idle", noise=0.05, seed=3)
    ui = UI(opts)
    ui.setup()
    try:
        ui._blit_right(None, None)
        food = ui._btn_rects["FOOD"].center
        step = ui._btn_rects["STEP"].center
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": food, "button": 1}))
        assert ui.events(), "clicking FOOD must not crash the event loop"
        assert ui.sim.last_button == "FOOD", "clicking the FOOD rect must fire the injector"
        was = ui.step_mode
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": step, "button": 1}))
        assert ui.events(), "clicking STEP must not crash the event loop"
        assert ui.step_mode == (not was), "clicking STEP toggles STEP mode"
    finally:
        pygame.quit()
    print("  headless: clicking FOOD fires injector; clicking STEP toggles mode")
    return True


TESTS = {k: v for k, v in globals().items() if k.startswith("test") and callable(v)}


def main(argv=None):
    which = argv[1:] if argv else []
    names = sorted(TESTS)
    if which:
        names = [n for n in names if any(w in n for w in which)]
    fails = 0
    for name in names:
        try:
            TESTS[name]()
            print(f"[PASS] {name}")
        except Exception as e:
            fails += 1
            import traceback

            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
    print(f"\n{len(names) - fails}/{len(names)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main(sys.argv)