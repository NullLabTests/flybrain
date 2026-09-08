"""Paths and CLI options for FLYBOARD."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

from . import __version__

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "data", "cache")
OUT_DIR = os.path.join(ROOT, "data", "out")
PRESETS_PATH = os.path.join(ROOT, "presets.yaml")
GRAPH_NPZ = os.path.join(OUT_DIR, "flyboard_graph.npz")
HITS_JSON = os.path.join(OUT_DIR, "hits.json")

# Real MaleCNS v1.0 files on the public Janelia GCS bucket.
REMOTE_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0"
WEIGHTS_REMOTE = (
    REMOTE_BASE
    + "/connectome-data/flat-connectome/"
    + "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)
ANNOTATIONS_REMOTE = (
    REMOTE_BASE
    + "/connectome-data/flat-connectome/"
    + "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
NEUROTRANS_REMOTE = (
    REMOTE_BASE
    + "/connectome-data/flat-connectome/"
    + "body-neurotransmitters-male-cns-v1.0.feather"
)
SOMA_PREFIX = REMOTE_BASE + "/malecns-v1.0-soma-points"

CACHE_FILES = {
    "body-annotations-male-cns-v1.0-minconf-0.5.feather": 14483314,
    "body-neurotransmitters-male-cns-v1.0.feather": 43282834,
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather": 1051241946,
}

# Local cache file names (same as remote for feathers).
ANN_FEATHER = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NTR_FEATHER = "body-neurotransmitters-male-cns-v1.0.feather"
W_FEATHER = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
SOMA_DIR = "soma-points"
SOMA_INFO = os.path.join(SOMA_DIR, "info")
SOMA_SHARDS = {
    "by_id": os.path.join(SOMA_DIR, "by_id", "0.shard"),
    "by_rel_body": os.path.join(SOMA_DIR, "by_rel_body", "0.shard"),
    "by_rel_nucleus_id": os.path.join(SOMA_DIR, "by_rel_nucleus_id", "0.shard"),
    "by_spatial_level_0": os.path.join(SOMA_DIR, "by_spatial_level_0", "0.shard"),
}
SOMA_SHARD_SIZES_BYTES = {
    "by_id": 10244020,
    "by_rel_body": 8889825,
    "by_rel_nucleus_id": 9104622,
    "by_spatial_level_0": 2554533,
}


def sorted_superclass_order():
    order = [
        "optic",
        "CX",
        "sensory",
        "descending",
        "VNC",
        "motor",
        "endocrine",
        "other",
        "ascending",
    ]
    return order


def build_superclass_groups():
    """Inferenced groups for population meters / leaderboard superclasses."""
    groups = {
        "optic": superclass_contains(("ol_", "visual_", "optic")),
        "central_complex (CX)": superclass_equals(("CX", "ellipsoid", "fb", "nod")),
        "sensory": superclass_contains(("sensory", "sensory_")),
        "ascending": superclass_contains(("ascending", "ascending_")),
        "descending": superclass_contains(("descending", "descending_")),
        "VNC": superclass_contains(("vnc_", "VNC")),
        "motor": superclass_equals(("cb_motor", "vnc_motor")),
        "endocrine": superclass_contains(("endocrine", "ENS")),
        "other": None,
    }
    return groups


def superclass_contains(parts):
    return lambda s: s is not None and any(p in s for p in parts)


def superclass_equals(parts):
    return lambda s: s is not None and s in parts


def default_palette():
    """Colors keyed by the flattened superclass string."""
    import numpy as np

    p = {
        "ol_intrinsic": (0.85, 0.35, 0.98),
        "visual_projection": (0.35, 0.95, 0.95),
        "visual_centrifugal": (0.15, 0.65, 0.90),
        "cb_intrinsic": (0.95, 0.85, 0.25),
        "cb_sensory": (0.25, 0.98, 0.45),
        "cb_motor": (1.0, 0.45, 0.10),
        "cb_endocrine": (0.95, 0.55, 0.85),
        "cb_efferent": (0.90, 0.80, 0.40),
        "vnc_intrinsic": (0.60, 0.60, 0.95),
        "vnc_sensory": (0.55, 0.92, 0.70),
        "vnc_motor": (0.98, 0.30, 0.25),
        "vnc_efferent": (0.85, 0.70, 0.30),
        "vnc_tbc": (0.70, 0.90, 0.95),
        "vnc_sensory_tbc": (0.80, 0.95, 0.75),
        "vnc_endocrine": (1.00, 0.70, 0.80),
        "sensory_ascending": (0.40, 0.90, 1.00),
        "ascending_neuron": (0.45, 0.55, 1.00),
        "descending_neuron": (1.00, 0.55, 0.15),
        "sensory_descending": (0.95, 0.70, 0.40),
        "ascending_neuron": (0.40, 0.55, 1.00),
        "efferent_descending": (0.95, 0.60, 0.45),
        "efferent_ascending": (0.55, 0.45, 1.00),
        "ENS": (1.00, 0.90, 0.55),
        "visual_projection_tbc": (0.55, 0.95, 0.95),
        "sensory_ascending_tbc": (0.60, 0.90, 1.00),
        "descending_neuron_tbc": (1.00, 0.60, 0.25),
    }
    return p


def make_arg_parser(prog="flyboard"):
    parser = argparse.ArgumentParser(prog=prog, description="FLYBOARD — MaleCNS whole-CNS current-injection arcade")
    parser.add_argument("--lite", action="store_true", help="Use the 4096-cell LITE BRAIN subsample")
    parser.add_argument("--source", default="synth:idle",
                        help="source pane: synth:idle | synth:pattern | image:PATH | video:PATH")
    parser.add_argument("--cite", action="store_true", help="Print citations and exit")
    parser.add_argument("--build-only", action="store_true", help="Build the cache graph and exit (no UI)")
    parser.add_argument("--headless", action="store_true", help="Run with dummy video driver (no window)")
    parser.add_argument("--step", action="store_true", help="Force STEP mode (ticks only on demand)")
    parser.add_argument("--noise", type=float, default=0.06, help="Noise sigma added each tick")
    parser.add_argument("--hold-ms", type=float, default=250.0, help="Injection hold length in ms")
    parser.add_argument("--drive", type=float, default=0.5, help="Default DRIVE fraction 0..1")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    return parser


@dataclass
class Options:
    lite: bool = False
    source: str = "synth:idle"
    cite: bool = False
    build_only: bool = False
    headless: bool = False
    step: bool = False
    noise: float = 0.06
    hold_ms: float = 250.0
    drive: float = 0.5
    seed: int | None = None
    argv: list = field(default_factory=list)

    @classmethod
    def from_args(cls, parser) -> "Options":
        args = parser.parse_args()
        return cls(
            lite=args.lite,
            source=args.source,
            cite=args.cite,
            build_only=args.build_only,
            headless=args.headless,
            step=args.step,
            noise=args.noise,
            hold_ms=args.hold_ms,
            drive=args.drive,
            seed=args.seed,
            argv=sys.argv[1:],
        )