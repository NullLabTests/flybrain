"""Headless screenshot gallery for the README.

Renders the real MaleCNS UI (whole 166k-cell stage on a dummy display) at a few
meaningful moments and saves PNGs to docs/screens/. Usage:

    .venv/bin/python scripts/screenshot_gallery.py [OUT_DIR]
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

from flyboard.config import Options
from flyboard.sim import Sim
from flyboard.ui import UI

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "screens")


def run_ticks(ui: UI, n: int):
    for _ in range(n):
        ui.sim.step()
        ui.source.tick()


def shot(ui: UI, name: str):
    os.makedirs(OUT, exist_ok=True)
    ui.draw()
    pygame.display.flip()
    path = os.path.join(OUT, name)
    pygame.image.save(ui.screen, path)
    print(f"  saved {path}")
    return path


def burst(ui: UI, button: str, amp: float, pre: int = 5, hold_tick: int = 8):
    run_ticks(ui, pre)                       # settle after previous burst
    ui.press(button)
    run_ticks(ui, hold_tick)                 # inside the hold window (rings + glow)
    name = button.lower()
    shot(ui, f"02_{name}_burst.png")


def main():
    opts = Options(headless=True, source="synth:idle", noise=0.06, seed=2026, drive=0.5)
    ui = UI(opts)
    ui.setup()
    # warm the rolling baseline so the leaderboard has a clean 1 s window
    run_ticks(ui, 60)
    shot(ui, "01_idle.png")

    burst(ui, "FOOD", amp=0.225)
    burst(ui, "MATE", amp=0.225)
    burst(ui, "ORGASM", amp=0.9)

    run_ticks(ui, 40)
    ui.sim.die()
    ui.death_banner = True
    shot(ui, "06_death.png")

    ui.press("RESET")
    shot(ui, "07_reset.png")

    print("gallery complete ->", OUT)


if __name__ == "__main__":
    main()