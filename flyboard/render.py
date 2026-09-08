"""Software 3D point-cloud renderer (pure numpy) for the brain stage.

Each cell is a point; size/brightness comes from recent spike activity or |v|.
Colors come from measured superclass. The result is an additive RGB image the
UI blits straight to the window - nothing is pre-baked.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from .graph import REGION_OPTIC


class Camera:
    def __init__(self, center=None, dist=1.0):
        self.yaw = 0.6
        self.pitch = -0.35
        self.dist = 1.0
        self.fov = 1.0
        self.center = np.zeros(3, dtype=np.float64)
        self.cx = 0
        self.cy = 0

    def rotation(self):
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        # Rz(yaw) @ Rx(pitch)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
        return Rz @ Rx


def project(camera, xyz, W, H, scale):
    """Project 3D points to screen (u,v) and return visibility."""
    n = len(xyz)
    R = camera.rotation()
    c = np.asarray(camera.center, dtype=np.float64)
    p = (xyz.astype(np.float64) - c) @ R.T
    d = camera.dist
    u = W / 2 + camera.cx * W * 0.5 + (p[:, 0] / d) * camera.fov * scale
    v = H / 2 + camera.cy * H * 0.5 - (p[:, 1] / d) * camera.fov * scale
    vis = (np.isfinite(u) & np.isfinite(v) & (p[:, 2] > -0.5 * d))
    ui = np.clip(u, 0, W - 1)
    vi = np.clip(v, 0, H - 1)
    return u, v, ui, vi, vis


def render(graph, sim, camera, H, W, scale, hidden_mask=None,
           intensity_scale=1.0, inject_marker=True, pixel_scale=2.0):
    """Return (rgb uint8, proj, seen).  `hidden_mask` filters cells (slice mode)."""
    n = graph.n
    xyz = graph.xyz
    u, v, ui, vi, vis = project(camera, xyz, W, H, scale)
    if hidden_mask is not None:
        keep = ~hidden_mask
        vis &= keep
    # per-cell intensity: recent |v| and spike brightness
    vmag = np.abs(sim.v)
    vmax = float(np.max(vmag)) if n else 0.0
    intensity = np.zeros(n, dtype=np.float32)
    if vmax > 0:
        intensity = (vmag / vmax).astype(np.float32)
    intensity = np.minimum(1.0, intensity * intensity_scale)
    intensity[sim.spikes] = 1.0
    intensity *= np.clip(vis, 0, 1).astype(np.float32)

    flat = (vi.astype(np.int64) * W + ui.astype(np.int64))
    chans = np.zeros((3, H * W), dtype=np.float32)
    col = graph.colors  # (n,3) in 0..1
    idx = np.flatnonzero(intensity > 0)
    if idx.size:
        f = flat[idx]
        it = intensity[idx]
        for ch in range(3):
            np.add.at(chans[ch], f, col[idx, ch] * it)
    img = chans.reshape(3, H, W)
    # soft glow via separable box filter
    k = max(1, int(pixel_scale))
    if k > 1:
        wpat = np.ones(k, dtype=np.float32) / k
        img = uniform_filter(img, size=(1, 3, 3)) * 1.6
    img = np.transpose(img, (1, 2, 0))  # H,W,3
    gain = 255.0 * 2.2
    rgb = np.clip(img * gain, 0, 255).astype(np.uint8)

    # white ring around injected cells during the hold window
    if inject_marker and sim.inject_active and sim.inject_indices.size:
        _draw_rings(rgb, u, v, sim.inject_indices, (255, 255, 255))

    return rgb, (u, v, vis), intensity


def _draw_rings(rgb, u, v, idx, color, radius=5):
    H, W = rgb.shape[:2]
    from scipy.ndimage import binary_dilation as dilate

    ring = np.zeros((H, W), dtype=bool)
    uu = np.clip(np.round(u[idx]).astype(np.int64), 0, W - 1)
    vv = np.clip(np.round(v[idx]).astype(np.int64), 0, H - 1)
    ring[vv, uu] = True
    ring = dilate(ring, iterations=radius) & ~dilate(ring, iterations=max(1, radius - 1))
    rgb[ring] = color


def nearest_cell(u, v, vis, mouse):
    """Hit-test a projected cell near the mouse cursor."""
    if vis is None or not vis.size:
        return -1
    mx, my = mouse
    d = (u - mx) ** 2 + (v - my) ** 2
    best = int(np.argmin(d))
    if best < vis.size and vis[best] and d[best] < 900:
        return best
    return -1


def slice_hidden_mask(graph):
    return graph.region_codes == REGION_OPTIC