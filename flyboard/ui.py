"""FLYBOARD arcade UI (pygame).

3 panes:
  LEFT   source pane (synth pattern / image / video - no camera)
  CENTER brain stage: 3D soma cloud of the MaleCNS graph
  RIGHT  buttons + population meters + "who lit up" leaderboard

Footer disclaimer is drawn on every frame.
"""
from __future__ import annotations

import os
import time

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from . import (
    FOOTER_DISCLAIMER,
    MALECNS_PAPER,
    SIM_DT_S,
    ORGASM_ACTIVE_FRACTION_CAP,
)
from .config import Options, make_arg_parser
from .graph import Graph
from .resolve import resolve_all, run_resolve
from .sim import Sim
from .source import SourceField, PULSE_AMP
from .stats import compute_panel
from .render import Camera, nearest_cell, slice_hidden_mask

W_PANE = 320
C_PANE = 1720 - 320 - 470
R_PANE = 470

BUTTON_COLORS = {
    "FOOD": (220, 80, 90),
    "PAIN": (220, 120, 40),
    "LICK": (110, 160, 220),
    "MATE": (190, 90, 220),
    "ORGASM": (255, 190, 40),
    "DEATH": (90, 90, 90),
}

KEYMAP = {}
try:
    import pygame

    KEYMAP = {"1": "FOOD", "2": "PAIN", "3": "LICK", "4": "MATE", "5": "ORGASM", "d": "DEATH", "r": "RESET"}
except Exception:  # pragma: no cover
    pygame = None


def _font(size, bold=False):
    return pygame.font.SysFont("dejavusansmono,monospace,liberationmono", size, bold=bold)


class UI:
    def __init__(self, opts: Options):
        self.opts = opts
        self.graph = None
        self.sim = None
        self.source = None
        self.resolved = {}
        self.step_mode = opts.step
        self.proxy_banner = None
        self.death_banner = False
        self.slice_mode = False
        self.pinned = []
        self.saved_cam = Camera()
        self.frames = 0
        self.last_t = time.time()
        self.acc = 0.0
        self.panel = None
        self.cm_ready = False
        self._btn_rects = {}
        self._last_panel_tick = 0

    # ------------------------------------------------------------ setup
    def setup(self):
        g = self.graph = Graph.load(lite=self.opts.lite, seed=self.opts.seed)
        g.finalize()
        print("graph:", g.banner(), "cells:", g.n, "fallback positions:", f"{g.fallback_fraction:.1%}")
        resolved = resolve_all(g)
        self.resolved = resolved
        from .resolve import write_hits
        write_hits(g, resolved)
        self.sim = Sim(g, noise=self.opts.noise, seed=self.opts.seed or 1)
        self.source = SourceField(self.opts.source, g)
        self.source.attach_graph(g)
        cam = Camera()
        cam.center = g.xyz.mean(axis=0).astype(np.float64)
        cam.dist = float(np.linalg.norm(g.xyz_max - g.xyz_min)) * 0.85 if g.n else 1.0
        self.cam = cam
        self._pygame_init()

    def _pygame_init(self):
        if self.opts.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self.WP = C_PANE
        w = W_PANE + C_PANE + R_PANE
        h = 940
        pygame.display.set_caption("FLYBOARD — MaleCNS whole-CNS current injection")
        self.screen = pygame.display.set_mode((w, h))
        self.font = _font(15)
        self.font_b = _font(15, bold=True)
        self.font_s = _font(12)
        self.font_t = _font(17, bold=True)
        self.font_big = _font(46, bold=True)
        pygame.display.flip()

    # ------------------------------------------------------------ buttons
    def press(self, name):
        if name == "DEATH":
            self.sim.die()
            self.death_banner = True
            self.proxy_banner = None
            return
        if name == "RESET":
            self.sim.reset(seed=self.opts.seed or None)
            self.death_banner = False
            self.proxy_banner = None
            self.pinned = []
            return
        if self.death_banner:
            self.press("RESET")
        r = self.resolved.get(name)
        if r is None:
            return
        hold_ms = float(r.report.get("hold_ms", self.opts.hold_ms))
        amp = self.opts.drive * 0.45
        extra_noise = bool(r.report.get("extra_noise", False))
        bounded = bool(r.report.get("bounded_extra_drive", False))
        if self.step_mode:
            self.sim.flush(8)  # warm the window so baseline is not empty
        self.sim.inject(r.indices, amp, hold_ms, extra_noise=extra_noise,
                        bounded_extra=bounded, button=name)
        if self.step_mode:
            # STEP mode: play out the burst (~1 s = hold + 300 ms measure) fast,
            # then the cloud freezes in its lit state.
            self.sim.flush(int(1.0 / SIM_DT_S))
        if r.proxy_used:
            sup = r.report.get("proxy_superclasses")
            self.proxy_banner = f"PROXY: no exact types, driving superclass {','.join(sup)}"
        else:
            self.proxy_banner = None
        if name == "ORGASM":
            self.proxy_banner = None
            self.orgasm_banner = time.time()
            self.panel = None

    def tick_sim(self):
        if self.death_banner:
            self.sim.step()
            return
        if self.step_mode:
            # NO continuous stepping in STEP mode: cloud stays frozen until a button
            return
        dt_now = time.time()
        self.acc += min(0.25, dt_now - self.last_t)
        self.last_t = dt_now
        steps = 0
        while self.acc >= SIM_DT_S and steps < 8:
            r = self.source.tick()
            self.sim.set_retina(r)
            self.sim.step()
            self.acc -= SIM_DT_S
            steps += 1
        if self.opts.step is False and self.sim.rt > 0 and self.frames > 240 and not self.step_mode:
            if self.sim.rt < 0.25:
                self.step_mode = True
                print("STEP MODE: real-time factor < 0.25; running full-capacity bursts on demand")

    # ------------------------------------------------------------ render
    def draw(self):
        sc = self.screen
        sc.fill((14, 14, 20))
        scale = 1.0
        rgb_src, _, _ = self._render_source()
        self._blit_left(rgb_src)
        rgb, proj, intens = self._render_brain()
        self._blit_center(rgb)
        self._blit_right(proj, intens)
        self._draw_footer()
        self.frames += 1

    def _render_source(self):
        img = self.source.frame_img if hasattr(self.source, "frame_img") else self.source.next_frame_img()
        img = np.clip(img, 0, 1)
        h, w = img.shape[:2]
        if len(self.source.optic_cells) and self.source.optic_cells.size == h * w and hasattr(self.sim, "retina"):
            drive = self.sim.retina[self.source.optic_cells].reshape(h, w)
            img = np.clip(img * 0.6 + drive[:, :, None] * 8.0, 0, 1)
        return (img * 255).astype(np.uint8), None, None

    def _render_brain(self):
        from . import render as rd

        hidden = slice_hidden_mask(self.graph) if self.slice_mode else None
        H = self.h_c = 940 - 46
        W = self.w_c = C_PANE - 20
        scale = 0.42 * W
        rgb, (u, v, vis), intens = rd.render(
            self.graph, self.sim, self.cam, H, W, scale=scale,
            hidden_mask=hidden, intensity_scale=2.5,
        )
        self.proj = (u, v, vis)
        return rgb, (u, v, vis), intens

    def _blit_left(self, rgb_src):
        x = 6
        y = 8
        label = self.source.kind
        self.screen.blit(self._as_surf(self.font_s.render("SOURCE  " + label, True, (160, 160, 180))), (x + 6, y))
        if rgb_src is not None:
            srf = pygame.surfarray.make_surface(np.ascontiguousarray(np.transpose(rgb_src, (1, 0, 2))))
            target = pygame.transform.scale(srf, (W_PANE - 16, W_PANE - 16))
            self.screen.blit(target, (x + 6, y + 22))
        self.text((x + 6, y + (W_PANE - 16) + 30), "eye input (spice)", (120, 200, 220))
        self.text((x + 6, y + (W_PANE - 16) + 48), "no camera module", (110, 110, 120))

    def _as_surf(self, r):
        return r

    def _blit_center(self, rgb):
        x0 = W_PANE + 4
        y0 = 6
        if rgb is not None:
            srf = pygame.surfarray.make_surface(np.ascontiguousarray(np.transpose(rgb, (1, 0, 2))))
            self.screen.blit(srf, (x0, y0))
        # banners
        if self.death_banner:
            t = self._txt("DEAD", (200, 60, 60), 46)
            self.screen.blit(t, (x0 + self.w_c // 2 - t.get_width() // 2, y0 + 150))
        self.text((x0 + 4, y0 + 2), self.graph.banner() + ("  ·  FALLBACK POS %.0f%%" % (self.graph.fallback_fraction * 100) if self.graph.fallback_fraction > 0.01 else ""), (235, 235, 235), bold=True)
        self.text((x0 + 4, y0 + 20), f"tick {self.sim.tick}   rt x{self.sim.rt:.2f}   |v| max {np.max(np.abs(self.sim.v)):.1f}   spikes {int(self.sim.spikes.sum())}", (170, 170, 190))
        if self.proxy_banner:
            self._banner_box((x0 + 6, y0 + 44), self.proxy_banner, (255, 150, 60))
        if getattr(self, "orgasm_banner", None) and time.time() - self.orgasm_banner < 6:
            self._banner_box((x0 + 6, y0 + 70), "FAKE REWARD BURST", (255, 190, 40))
            self.text((x0 + 6, y0 + 94), f"bounded drive only on the DA set · active-fraction cap {ORGASM_ACTIVE_FRACTION_CAP:.0%}", (200, 160, 80))
        if self.step_mode:
            self._banner_box((x0 + 6, y0 + 116), "STEP MODE: burst on button, then frozen lit cloud", (140, 140, 255))
        if self.slice_mode:
            self._banner_box((x0 + 6, y0 + 140), "SLICE: optic lobes hidden", (120, 200, 200))
        if self.pinned:
            py = y0 + self.h_c + 4
            self.text((x0, py), "pins:", (140, 200, 140))
            for pin in self.pinned[-6:]:
                py += 16
                self.text((x0 + 66, py), pin, (210, 230, 210))

    def _txt(self, s, color, size):
        f = _font(size, bold=True)
        return f.render(s, True, color)

    def _banner_box(self, pos, text, color):
        s = self.font_b.render(text, True, color)
        pad = 6
        box = pygame.Rect(pos[0], pos[1], s.get_width() + 2 * pad, s.get_height() + 6)
        bg = pygame.Surface((box.width, box.height), pygame.SRCALPHA)
        bg.fill((*[int(c * 0.18) for c in color], 220))
        self.screen.blit(bg, box)
        self.screen.blit(s, (pos[0] + pad, pos[1] + 3))

    def _blit_right(self, proj, intens=None):
        x0 = W_PANE + C_PANE + 6
        x1 = x0 + R_PANE - 12
        self.text((x0, 8), "CURRENT INJECTORS", (220, 220, 230), bold=True)
        y = 26
        btn_w = (R_PANE - 20) // 3
        for i, name in enumerate(["FOOD", "PAIN", "LICK", "MATE", "ORGASM", "DEATH", "RESET", "STEP", "SLICE"]):
            col = self._button_color(name)
            r = pygame.Rect(x0 + (i % 3) * (btn_w + 6), y + (i // 3) * 40, btn_w, 34)
            pygame.draw.rect(self.screen, col, r, border_radius=6)
            pygame.draw.rect(self.screen, (40, 40, 50), r, width=1, border_radius=6)
            label = {"STEP": "STEP MODE", "SLICE": "SLICE"}.get(name, name)
            t = self.font_b.render(label, True, (20, 20, 25))
            if name == "STEP":
                t = self.font_b.render(label + ("ON" if self.step_mode else "OFF"), True, (20, 20, 25))
            self.screen.blit(t, (r.x + (r.width - t.get_width()) // 2, r.y + 9))
            self._btn_rects[name] = r
        y += 3 * 40 + 8
        self.text((x0, y), f"DRIVE  {int(self.opts.drive * 100)}", (200, 200, 210))
        y += 18
        sl = pygame.Rect(x0, y, R_PANE - 16, 12)
        self._slider_rect = sl
        pygame.draw.rect(self.screen, (70, 70, 90), sl, border_radius=6)
        pygame.draw.rect(self.screen, (255, 190, 40), pygame.Rect(x0, y, max(2, int(sl.w * self.opts.drive)), 12), border_radius=6)
        y += 26
        self.text((x0, y), f"HOLD {self.opts.hold_ms:.0f} ms" , (200, 200, 210))
        y += 6
        self.text((x0, y), "inject amp: %.3f" % (self.opts.drive * 0.45), (200, 200, 210))
        y += 26
        pygame.draw.line(self.screen, (70, 70, 90), (x0, y), (x1, y))
        y += 8
        self._draw_meters(x0, x1, y)
        self._draw_leaderboard(x0, x1)

    def _click_button(self, pos):
        """Right-panel hit-test: return the pressed button name or None."""
        for name, r in getattr(self, "_btn_rects", {}).items():
            if r.collidepoint(pos):
                if name in ("STEP", "SLICE"):
                    toggle = {"STEP": "step_mode", "SLICE": "slice_mode"}[name]
                    setattr(self, toggle, not getattr(self, toggle))
                    return None
                return name
        sl = getattr(self, "_slider_rect", None)
        if sl is not None and sl.collidepoint(pos):
            self.opts.drive = min(1.0, max(0.0, (pos[0] - sl.x) / sl.w))
            return None
        return None

    def _button_color(self, name):
        if name == "RESET":
            return (70, 120, 200)
        if name in ("STEP", "SLICE"):
            base = (90, 90, 120)
            if name == "STEP" and self.step_mode:
                base = (150, 150, 255)
            if name == "SLICE" and self.slice_mode:
                base = (120, 200, 200)
            return base
        return BUTTON_COLORS.get(name, (120, 120, 120))

    def _draw_meters(self, x0, x1, y):
        g = self.graph
        self.text((x0, y), "POPULATION METERS (active fraction / <|v|>)", (170, 170, 190))
        y += 16
        for name, idx in list(g.groups.items()):
            if len(idx) == 0:
                continue
            v, frac = self.sim.group_state(idx)
            bar = pygame.Rect(x0, y, int((x1 - x0) * 0.55), 8)
            pygame.draw.rect(self.screen, (55, 55, 70), bar, border_radius=3)
            fill = pygame.Rect(x0, y, int(bar.w * min(1.0, frac * 30)), 8)
            pygame.draw.rect(self.screen, (150, 230, 150), fill, border_radius=3)
            self.text((x0 + bar.w + 8, y - 3), f"{name:11s} {frac:5.1%}  {v:5.2f}", (200, 200, 210))
            y += 13
            if y > 720:
                return

    def _draw_leaderboard(self, x0, x1):
        p = self.panel
        y0 = 460
        self.text((x0, y0), "WHO LIT UP  (extra spikes vs 1 s baseline)", (240, 240, 240), bold=True)
        y = y0 + 18
        if p is None:
            return
        em = "measuring…" if p.measure_active else f"burst {p.button or ''}   total extra {p.total_extra:.0f}"
        self.text((x0, y), em, (180, 180, 200))
        y += 16
        for name, val in p.top_types[:10]:
            self._row(x0, x1, y, f"{name[:26]:26s}", val)
            y += 14
        self.text((x0, y), "top superclasses:", (170, 170, 190))
        y += 15
        for name, val in p.top_super[:8]:
            self._row(x0, x1, y, name[:26], val)
            y += 14
        y += 4
        self._sparkline(x0, x1, y, p.sparkline)
        y += 40
        seed = " ".join(f"{n}×{c}" for n, c in p.seed_types[:4])
        ds = " ".join(f"{n}" for n, _ in p.downstream_types[:6])
        self.text((x0, y), f"SEED: {seed[:60]}", (255, 220, 220))
        y += 14
        self.text((x0, y), f"DOWNSTREAM ({p.n_downstream_cells}): {ds[:60]}", (220, 220, 255))

    def _row(self, x0, x1, y, name, val):
        self.text((x0, y), name, (200, 205, 215))
        self.text((x0 + 210, y), f"{val:+.1f}", (240, 240, 245))

    def _sparkline(self, x0, x1, y, spark):
        self.text((x0, y), "sparkline (total spikes/tick)", (170, 170, 190))
        w = x1 - x0
        pts = []
        for i, s in enumerate(spark):
            pts.append((x0 + i / 256.0 * w, y + 26 - min(26, float(s) / 200.0)))
        if len(pts) > 1:
            pygame.draw.lines(self.screen, (150, 220, 255), False, pts, 2)

    def _draw_footer(self):
        lines = FOOTER_DISCLAIMER.splitlines()
        for i, ln in enumerate(lines):
            t = self.font_s.render(ln, True, (120, 120, 140))
            self.screen.blit(t, (6, 940 - 24 + i * int(self.font_s.get_linesize())))

    def text(self, pos, s, color, bold=False):
        f = self.font_b if bold else self.font
        self.screen.blit(f.render(s, True, color), pos)

    # ------------------------------------------------------------ events
    def events(self):
        for ev in pygame.event.get():
            if os.environ.get("FLYBOARD_LOG_QUIT") and ev.type in (pygame.QUIT, pygame.KEYDOWN):
                print(f"[ui] event -> {ev}")
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                name = KEYMAP.get(pygame.key.name(ev.key))
                if name:
                    self.press(name)
                    continue
                if ev.key == pygame.K_s:
                    self.slice_mode = not self.slice_mode
                if ev.key == pygame.K_p:
                    self.step_mode = not self.step_mode
                if ev.key == pygame.K_EQUALS or ev.key == pygame.K_KP_PLUS:
                    self.opts.drive = min(1.0, self.opts.drive + 0.1)
                if ev.key == pygame.K_MINUS or ev.key == pygame.K_KP_MINUS:
                    self.opts.drive = max(0.0, self.opts.drive - 0.1)
            if ev.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_down = True
                self.mx0, self.my0 = ev.pos
                if ev.button == 4:
                    self.cam.dist *= 1.05
                elif ev.button == 5:
                    self.cam.dist *= 0.95
                elif ev.button == 1:
                    clicked = self._click_button(ev.pos)
                    if clicked is not None:
                        self.press(clicked)
                    elif hasattr(self, "proj") and self.proj[2] is not None and self.proj[2].size:
                        # brain stage is drawn at (W_PANE+4, 6) inside the window;
                        # projection coords are local to the stage bitmap.
                        stage_pos = (ev.pos[0] - (W_PANE + 4), ev.pos[1] - 6)
                        if stage_pos[0] >= 0 and stage_pos[0] < C_PANE - 20 and stage_pos[1] >= 0 and stage_pos[1] < self.h_c:
                            hit = nearest_cell(*self.proj[:2], self.proj[2], stage_pos)
                            if hit >= 0:
                                nm = self.graph.type_name_of(hit)
                                side = self.graph.side_names[int(self.graph.side[hit])]
                                sc = self.graph.super_name_of(hit)
                                self.pinned.append(f"{nm}  {sc}  side {side}")
                                if len(self.pinned) > 30:
                                    self.pinned = self.pinned[-30:]
            if ev.type == pygame.MOUSEBUTTONUP:
                self.mouse_down = False
            if ev.type == pygame.MOUSEMOTION and getattr(self, "mouse_down", False):
                dx, dy = ev.rel
                self.cam.yaw += dx * 0.005
                self.cam.pitch += dy * 0.005
                self.cam.pitch = max(-1.4, min(1.4, self.cam.pitch))
        return True

    def run(self):
        self.setup()
        self._btn_rects = {}
        pg_next = 0
        while self.events():
            self.tick_sim()
            if self.frames % 3 == 0 or self.panel is None:
                self.panel = compute_panel(self.graph, self.sim)
                self._last_panel_tick = self.sim.tick
            self.draw()
            pygame.display.flip()
            time.sleep(0.001)
        pygame.quit()
        print("bye")


def main():
    parser = make_arg_parser()
    opts = Options.from_args(parser)
    if opts.cite:
        from .cite import print_cite
        print_cite()
        return 0
    if opts.build_only:
        from .build import build
        build()
        return 0
    ui = UI(opts)
    ui.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())