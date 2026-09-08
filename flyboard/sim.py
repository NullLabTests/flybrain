"""Leaky-integrator simulation of the MaleCNS graph.

Per 20 ms tick (dt):
      v ← exp(-dt/0.1)*v + SIM_W_GAIN*W*spikes + SIM_BG + noise + retina + I_button
      spike if v >= 1, then v = 0

(SIM_W_GAIN ships 1.0 / SIM_BG ships 0.110 -- the same real MaleCNS matrix is
globally bistable at the canonical 1.5/0.180, so the shipped constants keep the
whole-brain sim in the near-critical regime. See flyboard/__init__.py.)

Butons inject I_button only into the resolved list of cell IDs; the glow you
see is the real voltage/time course from this update, not a canned animation.
"""
from __future__ import annotations

import time

import numpy as np

from . import SIM_DT_S, SIM_LEAK, SIM_W_GAIN, SIM_BG, THRESHOLD, ORGASM_ACTIVE_FRACTION_CAP
from .config import Options

try:
    from numba import njit

    @njit(cache=True, nogil=True)
    def _propagate(indptr, indices, data, spiking, v, gain):
        for j in range(spiking.size):
            s = spiking[j]
            for k in range(indptr[s], indptr[s + 1]):
                v[indices[k]] += gain * data[k]

    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False

    def _propagate(indptr, indices, data, spiking, v, gain):  # pragma: no cover
        lens = indptr[spiking + 1] - indptr[spiking]
        total = int(lens.sum())
        if total == 0:
            return
        offs = np.zeros(len(lens) + 1, dtype=np.int64)
        offs[1:] = lens.cumsum()
        start = indptr[spiking]
        edges = np.repeat(start, lens) + (np.arange(total) - np.repeat(offs[:-1], lens))
        np.add.at(v, indices[edges], gain * data[edges])


class Sim:
    def __init__(self, graph, opts: Options | None = None, rng=None, dt=SIM_DT_S,
                 noise=0.06, seed=None):
        self.g = graph
        n = graph.n
        W = graph.W.tocsr()
        self.indptr = W.indptr.astype(np.int64)
        self.indices = W.indices.astype(np.int64)
        self.data = W.data.astype(np.float32)
        self.n = n
        self.dt = dt
        self.tau = 0.1
        self.leak = np.float32(np.exp(-dt / self.tau))
        self.gain = np.float32(SIM_W_GAIN)
        self.bg = np.float32(SIM_BG)
        self.thr = np.float32(THRESHOLD)
        self.noise_sigma = float(noise)
        self._rng = rng or np.random.default_rng(seed)
        self.v = np.zeros(n, dtype=np.float32)
        self.spikes = np.zeros(n, dtype=bool)
        self.tick = 0
        self.total_spikes = 0

        # rolling 1 s baseline (per-tick ring, 50 ticks)
        self.window = 50
        self.ring = np.zeros((self.window, n), dtype=np.uint8)
        self.ring_pos = 0
        self.window_sum = np.zeros(n, dtype=np.float32)

        self.retina = np.zeros(n, dtype=np.float32)
        self.dead = False

        # injection state
        self.inject_active = False
        self.inject_set = np.zeros(n, dtype=bool)
        self.inject_indices = np.empty(0, dtype=np.int64)
        self.I_button = np.zeros(n, dtype=np.float32)
        self.hold_ticks_left = 0
        self.extra_noise = np.zeros(n, dtype=np.float32)
        self.last_button = None
        self.drive_amp = 0.0
        self.bounded_extra = False

        # burst measure window (hold + 300 ms)
        self.measure_active = False
        self.measure_left = 0
        self.measure_total = 0
        self.meas = np.zeros(n, dtype=np.float32)
        self.base_snapshot = np.zeros(n, dtype=np.float32)
        self.burst_spikes = np.zeros(0, dtype=np.float32)

        # sparkline history (total spikes per tick, ring of 256)
        self.spark = np.zeros(256, dtype=np.int64)
        self.spark_pos = 0

        self.rt = 1.0  # real-time factor, refreshed each tick

    # ------------------------------------------------------------ physics
    def step(self):
        t0 = time.perf_counter()
        n = self.n

        if self.dead:
            self.v.fill(0.0)
            self.spikes[:] = False
            new_spikes = 0
        else:
            v = self.v
            v *= self.leak
            v += self.bg
            v += self._rng.standard_normal(n).astype(np.float32) * self.noise_sigma
            v += self.retina
            v += self.I_button
            v += self.extra_noise
            sp = np.flatnonzero(self.spikes)
            if sp.size:
                _propagate(self.indptr, self.indices, self.data, sp, v, self.gain)
            v = np.minimum(v, 4.0)
            new_spk = v >= self.thr
            v[new_spk] = 0.0
            self.v = v
            self.spikes = new_spk
            new_spikes = int(new_spk.sum())
            self.total_spikes += new_spikes

        # ring baseline
        old = self.ring[self.ring_pos]
        self.ring[self.ring_pos] = self.spikes.astype(np.uint8)
        self.window_sum += self.spikes.astype(np.float32) - old.astype(np.float32)
        self.ring_pos = (self.ring_pos + 1) % self.window

        self.spark[self.spark_pos] = new_spikes
        self.spark_pos = (self.spark_pos + 1) % self.spark.size

        self.tick += 1
        self._tick_timers(new_spikes)

        dt_wall = time.perf_counter() - t0
        if dt_wall > 0:
            self.rt = (self.dt / dt_wall)
        return new_spikes

    def _tick_timers(self, new_spikes):
        if self.inject_active:
            self.hold_ticks_left -= 1
            if self.hold_ticks_left <= 0:
                self.end_injection()
        if self.measure_active:
            self.meas += self.spikes.astype(np.float32)
            idx = self.measure_total - self.measure_left
            if 0 <= idx < self.burst_spikes.size:
                self.burst_spikes[idx] = new_spikes
            self.measure_left -= 1
            if self.measure_left <= 0:
                self.measure_active = False

    # ------------------------------------------------------------ injection
    def inject(self, indices, amp, hold_ms, extra_noise=False, bounded_extra=False,
               button=None):
        if self.dead:
            return
        indices = np.asarray(indices, dtype=np.int64)
        self.end_injection()
        if bounded_extra:
            # FAKE REWARD BURST: drive at most cap*n reward cells so the reward
            # itself can never flood the net (MaleCNS idle sits ~0.007 V below
            # threshold because bg=0.180, tau=0.1 -> v_ss ~= 0.993).
            capn = max(1, int(round(ORGASM_ACTIVE_FRACTION_CAP * self.n)))
            if len(indices) > capn:
                rng = np.random.default_rng(2026)
                pick = rng.choice(len(indices), size=capn, replace=False)
                indices = np.sort(indices[pick])
        self.inject_indices = indices
        self.inject_set = np.zeros(self.n, dtype=bool)
        self.inject_set[indices] = True
        self.I_button.fill(0.0)
        self.I_button[indices] = amp
        self.drive_amp = amp
        self.hold_ticks_left = max(1, int(round(hold_ms / (self.dt * 1000))))
        self.inject_active = True
        self.last_button = button
        self.bounded_extra = bounded_extra
        if extra_noise:
            self.extra_noise[indices] = self.noise_sigma * 1.5
        # start the burst measurement (hold + 300 ms after hold)
        self.measure_total = self.hold_ticks_left + int(round(0.3 / self.dt))
        self.burst_spikes = np.zeros(self.measure_total, dtype=np.float64)
        self.measure_left = self.measure_total
        self.measure_active = True
        self.meas.fill(0.0)
        self.base_snapshot = self.window_sum * (self.measure_total / self.window)

    def end_injection(self):
        self.I_button.fill(0.0)
        self.extra_noise.fill(0.0)
        self.inject_active = False
        self.inject_set[:] = False

    # ------------------------------------------------------------ modes
    def die(self):
        self.dead = True
        self.v.fill(0.0)
        self.spikes[:] = False
        self.I_button.fill(0.0)
        self.extra_noise.fill(0.0)
        self.inject_active = False
        self.inject_set[:] = False

    def reset(self, seed=None):
        self.dead = False
        self._rng = np.random.default_rng(seed)
        self.v.fill(0.0)
        self.spikes[:] = False
        self.I_button.fill(0.0)
        self.extra_noise.fill(0.0)
        self.inject_active = False
        self.inject_set[:] = False
        self.measure_active = False
        self.ring[:] = 0
        self.window_sum.fill(0.0)
        self.ring_pos = 0
        self.spark[:] = 0
        self.spark_pos = 0
        self.meas.fill(0.0)
        self.base_snapshot.fill(0.0)
        self.burst_spikes = np.zeros(0, dtype=np.float32)
        self.tick = 0
        self.total_spikes = 0
        self.note = None

    def set_retina(self, drive):
        self.retina[:] = np.asarray(drive, dtype=np.float32).ravel()

    def flush(self, n_ticks=60):
        """STEP mode: run `n_ticks` as fast as possible."""
        for _ in range(n_ticks):
            self.step()

    # ------------------------------------------------------------ meters
    def group_state(self, group_idx):
        if len(group_idx) == 0:
            return 0.0, 0.0
        v = np.abs(self.v[group_idx]).mean()
        frac = float(self.spikes[group_idx].mean())
        return float(v), frac

    def active_fraction(self):
        return float(self.spikes.mean())

    def reward_driven_fraction(self):
        """Fraction of the net driven by the current bounded reward burst
        (guaranteed <= ORGASM_ACTIVE_FRACTION_CAP by construction)."""
        if not self.bounded_extra or not len(self.inject_indices):
            return 0.0
        return float(len(self.inject_indices) / max(1, self.n))

    def average_abs_v(self):
        return float(np.abs(self.v).mean())

    def injected_hold_ms_remaining(self):
        return self.hold_ticks_left * self.dt * 1000 if self.inject_active else 0.0

    def leaderboard(self, top_types=15, top_super=8):
        """Aggregate the active burst measure (hold + 300 ms)."""
        extra = self.meas - self.base_snapshot
        return extra

    def seed_downstream(self):
        extra = self.lead_extra if hasattr(self, "lead_extra") else self.leaderboard()
        seed = set(int(i) for i in self.inject_indices)
        ds = [int(i) for i in np.flatnonzero(extra > 0) if i not in seed]
        return seed, ds