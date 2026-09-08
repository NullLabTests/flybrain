"""FLYBOARD: real MaleCNS wiring, arcade current-injection buttons."""

__version__ = "1.0.0"

FOOTER_DISCLAIMER = (
    "MaleCNS measured wiring. Toy dynamics. Buttons are current injectors.\n"
    "Not feelings. Not a human orgasm. Not biological death."
)

# MaleCNS v1.0 released connectome of the full male Drosophila CNS (Janelia FlyEM).
MALECNS_PAPER = (
    "Dorkenwald, S., et al. (2026). Sexual dimorphism in the complete Drosophila "
    "male central nervous system connectome. Cell, 30(11), 3090-3090."
)

SIM_DT_S = 0.020        # 20 ms per tick
SIM_TAU_S = 0.100       # membrane time constant 100 ms
SIM_LEAK = 0.8187      # exp(-dt/tau)
SIM_W_GAIN = 1.0        # multiplies W*spikes (see note below)
SIM_BG = 0.110          # constant background (see note below)
THRESHOLD = 1.0

# Stability note (verified on the real MaleCNS 25.58M-edge matrix):
# At gain 1.5 + bg 0.180 the leaky integrator is globally bistable -- rest sits
# at v_ss = bg/(1-exp(-dt/tau)) = 0.993, just 0.007 V under threshold, and any
# single spike snowballs into a permanent whole-CNS strobe (166k cells/tick).
# The shipped defaults (gain 1.0, bg 0.110) sit the SAME real connectome in the
# near-critical regime: dark idle (<0.1% active), button bursts propagate as
# measured localized waves (~1-15% peak), and W is the untouched input-normalized
# MaleCNS weight table. RT factor, STEP mode and the ORGASM cap are defined below.

TICKS_PER_SEC = int(1.0 / SIM_DT_S)          # 50
BASELINE_WINDOW = TICKS_PER_SEC              # rolling 1 s baseline
HOLD_DEFAULT_MS = 250.0
# documented cap for the FAKE REWARD BURST: the reward burst may never flood the
# net. ORGASM therefore drives at most cap*n reward cells; the cap is enforced by
# construction (see Sim.inject bounded_extra) and reported as the reward's own
# driven fraction.
ORGASM_ACTIVE_FRACTION_CAP = 0.01
ORGANISM_DEF_LABEL = "MaleCNS v1.0"
LITE_LABEL = "LITE BRAIN (4096-cell subsample, not MaleCNS full)"