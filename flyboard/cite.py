"""--cite output: honest attribution for FLYBOARD."""
from __future__ import annotations

from . import FOOTER_DISCLAIMER, MALECNS_PAPER

CITE_TEXT = f"""FLYBOARD citations
==================

Primary connectome
------------------
{MALECNS_PAPER}

This is the finished connectome of the entire male Drosophila CNS
(central brain + both optic lobes + ventral nerve cord), released as
MaleCNS v1.0 (Janelia FlyEM / Cambridge / Google Research). FLYBOARD uses
the real released flat-connectome weight table (synapse-strength directed
edges), the curated body annotations, and the measured soma-point shards.

Data download:  https://male-cns.janelia.org/download
Data license:   CC-BY

Dynamics
--------
Leaky integrator, dt = 20 ms, tau = 100 ms:
    v <- exp(-dt/tau)*v + SIM_W_GAIN*W*spikes + SIM_BG + noise + retina + I_button
Leaky integrator, dt = 20 ms, tau = 100 ms. Surface constants:
    SIM_W_GAIN = 1.0, SIM_BG = 0.110 (shipped). At the canonical 1.5 / 0.180 the
real MaleCNS matrix is globally bistable (rest sits 0.007 V under threshold and
any spike strobes the whole CNS), so the shipped constants run the same real
wiring in its near-critical regime: dark idle, localized measured waves.
Weights are measured directed synapse counts, input-normalized at build time.

Buttons
-------
Buttons are current injectors (I_button) into real MaleCNS type names,
resolved against the loaded graph and reported in data/out/hits.json.
No canned animation is played; the glow is the sim's own voltage/spike field.

FOOD preset
-----------
Similar idea: drive taste (gustatory) sensory neurons and watch downstream
lights. Shiu et al., "Taste neurons in Drosophila..." (2024) mapped the
taste-to-feeding circuit; we do NOT reproduce their feeding circuit - we
only borrow the idea of "drive taste cells, watch downstream". Cite them if
you discuss feeding circuits:
    Shiu, P. K., Sterne, G. R., Spiller, N., Franconville, R., et al. (2024).
    A Drosophila computational brain model reveals sensorimotor processing.
    Nature 634, 870-877.  https://doi.org/10.1038/s41586-024-07763-9

ORGASM preset
-------------
`FAKE REWARD BURST` is a bounded current pulse into MaleCNS cells annotated
as dopaminergic / reward-associated (predicted transmitter dopamine). It is
not a biological reward event, and the active-fraction cap keeps the whole
net from being flooded.

{FOOTER_DISCLAIMER}
"""


def print_cite():
    print(CITE_TEXT)