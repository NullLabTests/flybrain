# FLYBOARD

**Real MaleCNS wiring. Toy dynamics. Buttons are current injectors.**

FLYBOARD is an arcade window into the **full 166,700-neuron male fruit-fly central
nervous system (MaleCNS v1.0, Janelia FlyEM)**. Six arcade buttons inject simulated
current directly into the *actual resolved cell populations* of the real connectome's
25.58M directed synapses. The glow on the brain is **not an animation** — it is the
live voltage/spike field of the leaky-integrator simulation crossing that real wiring.

| | | |
|---|---|---|
| cells | **166,700** neurons (whole CNS, typed + positioned) | |
| edges | **25,582,938** directed synapse counts (input-normalized) | |
| connectome | MaleCNS v1.0 (Dorkenwald et al. 2026) | |
| runtime | ~4x real-time on the full net, ~113x in `--lite` | |
| release | [MIT](./LICENSE) · Python 3.12+ · pygame-ce | |

![idle](docs/screens/01_idle.png)
*Idle — untouched MaleCNS wiring sits dark, near-critical.*

![food](docs/screens/02_food_burst.png)
*FOOD press — taste cells (`SNta*`) get injected; the wave is real voltage propagation.*

![mate](docs/screens/02_mate_burst.png)
*MATE press — the courtship circuit (pIP1 et al.) lights up.*

![orgasm](docs/screens/02_orgasm_burst.png)
*ORGASM — "FAKE REWARD BURST", bounded to a 1%-of-net dopamine subset.*

![death](docs/screens/06_death.png)
*DEATH — the whole brain freezes. Not biological death.*

---

## Honesty, first

> MaleCNS measured wiring. Toy dynamics. Buttons are current injectors.
> Not feelings. Not a human orgasm. Not biological death.

- The **connectome is real** (synapse-level, from the MaleCNS v1.0 release).
- The **dynamics are a toy**: a leaky integrator with synapse weights as
  conductances. It is tuned to sit **near-critical** — dark at rest, localized
  measured waves on demand — because the *same real matrix* is globally bistable
  at canonical settings (see *Physics*). None of this claims to be the fly's
  actual neurodynamics; it is honest play on real anatomy.
- There is **no camera module** anywhere in the codebase.

---

## What the buttons do

Buttons inject a constant `I_button` into the resolved MaleCNS cell IDs for a
`HOLD` window (default 250 ms). The glowing cloud **is** the voltage field.

| Button | Key | Drives these real MaleCNS cells | Effect |
|---|---|---|---|
| **FOOD** | `1` | taste cells, `SNta*` family prime (2799 cells, 57 types) | sweet/gustatory burst |
| **PAIN** | `2` | nociceptor class, includes ppk receptor cells (257 cells) | aversive localized wave |
| **LICK** | `3` | proboscis extension motor neurons (68 cells) | reflexive feeding motor burst |
| **MATE** | `4` | courtship circuitry incl. **pIP1** (3165 cells) | courtship ensemble lights |
| **ORGASM** | `5` | the real **dopaminergic** population (4443 cells) | **FAKE REWARD BURST** — capped, see below |
| **DEATH** | `d` | — | freezes every voltage to 0, whole brain goes dark |
| **RESET** | `r` | — | clears death, restores near-critical idle |

These counts come from `data/out/hits.json` written on every load, with
`requested_types` / `found_count` / `missing_types` / `proxy_used` per button.
If a button resolves to zero exact cells (e.g. on the LITE subsample) it still
fires as an orange **PROXY** by driving a superclass subset.

### FAKE REWARD BURST — the ORGASM cap

MaleCNS idle sits `SIM_BG` below threshold; a big drive can ignite the entire CNS.
The reward burst is therefore **bounded by construction**: ORGASM drives at most
`cap * n` reward cells (cap = **0.01**, i.e. ~1,667 of the DA population), enforced
in `Sim.inject(..., bounded_extra=True)` and reported live as `reward_driven_fraction`.

### Leaderboard ("WHO LIT UP")

Light-up score = *(spikes during hold + 300 ms)* − *(rolling 1 s baseline)*,
counting only cells that beat baseline by **≥ 1 spike**. Rendered per type and
per superclass, so a burst's identity is legible as fly types (`LC10a`,
`ORN_DA1`, `TmY21`, …), not just "something lit up".

---

## Physics

Per 20 ms tick:

```
v  <-  exp(-dt/0.1) * v  +  SIM_W_GAIN * W·spikes  +  SIM_BG  +  noise  +  retina  +  I_button
spike if v >= 1:  v = 0
```

`W` is the untouched MaleCNS weight table (synapse counts, column-normalized per
postsynaptic cell). Shipped constants: **`SIM_W_GAIN = 1.0`**, **`SIM_BG = 0.110`**,
noise `0.06`.

**Why not the "obvious" stronger drive?** At the canonical `1.5 / 0.180`, the real
25.58M-edge matrix is *globally bistable*: rest sits at `v_ss = 0.18/(1 − e^(−0.2)) ≈ 0.993`,
just 0.007 V under threshold — any single spike (even a noise blip) snowballs into
a permanent whole-CNS strobe (166k cells/tick). The shipped constants run the
**same untouched wiring** in the near-critical regime: dark idle, button bursts are
measured, localized waves, and a meaningful "who lit up" board.

---

## Run it

```bash
./run-flyboard fetch          # download ~1.3 GB of Janelia MaleCNS v1.0 tables + soma shards
./run-flyboard build          # build data/out/flyboard_graph.npz (166,700 cells) + manifest
./run-flyboard                # full 166k net, arcade UI
./run-flyboard --lite         # 4096-cell LITE BRAIN subsample (~113x real-time)
./run-flyboard --source synth:pattern   # moving eye input
./run-flyboard --source image:PATH      # or video:PATH
./run-flyboard --step         # force STEP mode (burst on button, then frozen cloud)
./run-flyboard --cite         # citation, then exits
.venv/bin/python tests/test_flyboard.py   # 9/9 spec tests
```

No MaleCNS cache? `--lite` falls back to a synthetic fixture so the UI always runs.

## Controls

| Input | Action |
|---|---|
| **1–5** / **d** / **r** | FOOD / PAIN / LICK / MATE / ORGASM / DEATH / RESET |
| **mouse click on panel button** | same as the key |
| **`+` / `−`** (or panel slider) | DRIVE ± 10% — how strong the next press is |
| **p** / **STEP** | STEP mode toggle (burst on button, then frozen lit cloud) |
| **s** / **SLICE** | hide the optic lobes |
| **drag** | rotate the 3D soma cloud |
| **mouse wheel** | zoom |
| **click a glowing cell** | pin it: type · superclass · side (up to 6 shown) |
| **Esc / q** | quit |

## Layout

```
flyboard/         python package
  __init__.py     constants, footer, ORGASM cap
  build.py        feather -> csr graph build (real data)
  soma.py         neuroglancer AnnotationReader -> measured soma positions
  data.py         fetch/status of the Janelia cache
  graph.py        load / LITE subsample / region codes / colors
  resolve.py      button -> MaleCNS type -> cell IDs (+ hits.json, proxies)
  sim.py          leaky-integrator physics, hold/measure windows
  source.py       eye input pane (synth / image / video)          [no camera]
  render.py       software 3D point-cloud renderer (pure numpy)
  stats.py        "who lit up" leaderboard
  ui.py           3-pane pygame UI
  cite.py         paper + data citations
tests/            headless spec tests (scripted clicks included)
scripts/          screenshot gallery generator
presets.yaml      button -> type/selector definitions
```

---

## Inspiration & provenance

FLYBOARD is a small love letter to the 2026 MaleCNS v1.0 release:

> Dorkenwald, S., et al. (2026). *Sexual dimorphism in the complete Drosophila male
> central nervous system connectome.* Cell, 30(11), 3090–3090.
> Data: Janelia Research Campus / Google Research FlyEM, Cambridge, and collaborators.

- **The anatomy is real** — every resolved button population is a typed MaleCNS cell
  group (taste `SNta*`, ppk nociceptors, proboscis motor neurons, pIP1 courtship
  neurons, the dopamine projection set).
- **Soma positions** are the measured MaleCNS soma-points shards, read via the
  neuroglancer annotation reader (no hand-rolled binary parsing).
- **The arcade idea** — reward buttons that light a *brain* — nods to the classic
  "pleasure button" thought experiment (Olds & Milner 1954) and to every demo where
  the glow is honestly computed rather than faked.

The fly deserves better than a strobe. FLYBOARD keeps the real wiring at its own
critical edge so a press shows you *which corner of the MaleCNS actually reacts*.

---

## License

[MIT](./LICENSE). Data remains © its respective owners (Janelia FlyEM et al.),
MIT license applies only to the FLYBOARD code.