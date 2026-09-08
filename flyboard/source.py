"""Source pane: synthetic eye input or image/video file. No camera, ever."""
from __future__ import annotations

import os

import numpy as np

GRID_H, GRID_W = 72, 72
IDLE_BASE = 0.015
PULSE_AMP = 0.20


class SourceField:
    def __init__(self, kind="synth:idle", graph=None, seed=11):
        self.kind = kind
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.h, self.w = GRID_H, GRID_W
        # per-cell driver mapping for optic+sensory cells
        self.optic_cells = None
        self.set_graph(graph)
        self.video = None
        self._open(kind)

    def set_graph(self, graph):
        if graph is None:
            self.optic_cells = np.arange(0)
            return
        g = graph
        m = np.zeros(g.n, dtype=bool)
        for nm, idx in g.groups.items():
            if nm == "optic":
                m[idx] = True
        cells = np.flatnonzero(m)
        if len(cells) < 1:
            # LITE subsamples may have no optic-lobe cells; fall back to sensory
            m[:] = False
            for nm, idx in g.groups.items():
                if nm == "sensory":
                    m[idx] = True
            cells = np.flatnonzero(m)
        if len(cells) < self.h * self.w:
            cells = np.concatenate([cells, np.zeros(self.h * self.w - len(cells), dtype=np.int64)])
        self.optic_cells = cells[: self.h * self.w]

    def _open(self, kind):
        if kind.startswith("image:"):
            try:
                from PIL import Image
                p = kind.split(":", 1)[1]
                im = Image.open(p).convert("RGB").resize((self.w, self.h))
                self.frames = [np.asarray(im, dtype=np.float32) / 255.0]
                self.is_frames = True
                self.kind = f"image:{os.path.basename(p)}"
                return
            except Exception as e:
                print(f"image source failed ({e}); using synth:pattern")
                self.is_frames = False
                self.kind = "synth:pattern"
                return
        if kind.startswith("video:"):
            try:
                import imageio.v2 as iio
                p = kind.split(":", 1)[1]
                self.video = iio.get_reader(p)
                self.kind = f"video:{os.path.basename(p)}"
                self.is_frames = True
                self.frames = [np.zeros((self.h, self.w, 3), np.float32)]
                return
            except Exception as e:
                print(f"video source failed ({e}); using synth:pattern")
                self.kind = "synth:pattern"
        self.is_frames = False

    def next_frame_img(self):
        """Return the (H, W, 3) display image in 0..1."""
        if self.is_frames and self.kind.startswith("image"):
            return self.frames[0]
        if self.is_frames and self.kind.startswith("video"):
            try:
                img = self.video.get_next_data()
            except Exception:
                self.video = None
                self.is_frames = False
                return self._synth()
            img = np.asarray(img, dtype=np.float32)
            if img.ndim == 2:
                img = np.repeat(img[:, :, None], 3, axis=2)
            img = img[: self.h, : self.w]
            pad_y = max(0, self.h - img.shape[0])
            pad_x = max(0, self.w - img.shape[1])
            img = np.pad(img, ((0, pad_y), (0, pad_x), (0, 0)))[: self.h, : self.w]
            mx = np.nanmax(img) or 1.0
            return img / mx
        return self._synth()

    def _synth(self):
        t = self.t
        rng = self.rng
        if self.kind == "synth:idle":
            drift = 0.15 * (np.sin(t / 40.0) + 1.0) / 2.0
            a = 0.9 * IDLE_BASE
            base = a * (0.7 + 0.3 * np.sin(2 * np.pi * np.arange(self.h)[:, None] / 9) *
                        np.sin(2 * np.pi * np.arange(self.w)[None, :] / 11))
            tw = rng.standard_normal((self.h, self.w)) * (a * 0.35)
            return np.clip(base + drift * a + tw, 0, 1)[:, :, None].repeat(3, axis=2)
        if self.kind == "synth:pattern":
            yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
            g = 0.5 + 0.5 * np.sin(2 * np.pi * (xx * 0.12 + t * 0.02))
            pulse = 0.0
            if t % 90 == 0:
                pulse = 1.0
            field = PULSE_AMP * pulse + 0.15 * g
            return np.clip(field, 0, 1)[:, :, None].repeat(3, axis=2)
        idle = (0.8 * IDLE_BASE + 0.2 * IDLE_BASE * rng.random((self.h, self.w)))
        return np.clip(idle, 0, 1)[:, :, None].repeat(3, axis=2)

    def tick(self):
        img = self.next_frame_img()
        self.frame_img = img
        gray = img.mean(axis=2)
        drive = np.zeros(self.optic_cells.size, dtype=np.float32)
        drive[:] = gray.ravel()[: self.optic_cells.size]
        g_n = getattr(self, "g_n", 0)
        if g_n <= 0 and self.optic_cells.size:
            g_n = int(self.optic_cells.max()) + 1
        ret = np.zeros(max(g_n, 0), dtype=np.float32)
        if self.optic_cells.size and int(self.optic_cells.max()) < ret.size:
            ret[self.optic_cells] = drive * PULSE_AMP
        self.t += 1
        return ret

    def attach_graph(self, graph):
        self.set_graph(graph)
        self.g_n = graph.n