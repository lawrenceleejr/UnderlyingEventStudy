# UnderlyingEventStudy

**Can the soft underlying event predict the longitudinal boost of the hard scatter?**

The longitudinal boost of a hard scatter is fixed by the colliding partons'
momentum fractions `x1, x2` (boost rapidity `y = ½ ln(x1/x2)`). The beam remnants
and the soft underlying event (UE) that accompany every collision are
kinematically tied to `x1, x2` (Sjöstrand–Skands beam-remnant model,
[hep-ph/0402078](https://arxiv.org/abs/hep-ph/0402078); forward-energy ↔ beam-x,
[ALICE 2107.10757](https://arxiv.org/abs/2107.10757)). This project tests, with a
machine-learning regressor, whether the **soft particles alone** encode the boost
of the Z in `Z→μμ` events — the Z boost is measured from the two muons, and the
network sees only the *non-muon* soft particles. The estimator is then
transferred to `W→μν`, where the longitudinal degree of freedom is unmeasured
(the W-mass motivation).

This is, to our knowledge, a novel "inverse" use of the underlying event: prior
work regresses the *MPI count* from soft observables
([Ortiz et al. 2020](https://arxiv.org/abs/2004.03800)) or `x1,x2` from the
*hard* final state ([Sborlini et al. 2112.05043](https://arxiv.org/abs/2112.05043)),
but not the hard-scatter longitudinal boost from the soft event.

**Results:** see [`FINDINGS.md`](FINDINGS.md) (physics conclusions),
`results/REPORT_opendata.md` (the real-data measurement and the muon-footprint
artifact it required catching), and `results/REPORT_wmass.md` (Z→W transfer).

## Run it

### Simulation study (runs anywhere — Pythia8)

```bash
bash train.sh          # pip install + train EFN+Transformer on the best device
```

Auto-uses Apple Metal (MPS) on an M-series Mac, CUDA on NVIDIA, else CPU.
A compact 35k-event dataset (`data/skim/pythia_demo.parquet`) is committed so
this trains immediately. Full manual flow:

```bash
pip install -r requirements.txt
bash sim/build_pythia.sh               # builds Pythia8+python (honors PYTHIA_PREFIX)
export PYTHONPATH=/opt/pythia8312/lib:$PWD
python -m sim.generate --nevents 200000 --out data/skim/pythia.parquet
python -m src.evaluate --parquet data/skim/pythia.parquet --models efn --tag pythia
python -m src.ablation --parquet data/skim/pythia.parquet    # central/forward dissection
```

### Real CMS collision data (runs anywhere — no CMSSW required)

The diffuse soft tracks live in MiniAOD's `packedPFCandidates`. **`src/miniaod.py`
decodes that packed format directly with uproot** (IEEE-half minifloats for
pT/m/dz, scaled int16 for η/φ, PV-association quality bits), so the full
soft-track measurement needs no CMSSW, no Docker — just network and disk:

```bash
# skim N files of DoubleMuon Run2016G MiniAOD (record 30505) -> parquet
python -m src.skim --miniaod --record 30505 --nfiles 20 --out data/skim/miniaod.parquet
# the measurement matrix: data vs simulation, charged-PV + full variants,
# baselines, EFN, and a shuffled-target leakage control
python -m src.measure --data data/skim/miniaod.parquet --sim data/skim/pythia.parquet
```

Z candidates come from the PF muons (validated: Z peak at 90.7 GeV); pileup is
suppressed for charged candidates via the leading-PV association; the muon
footprint is removed with a wide ΔR<0.4 veto (mandatory on detector data — see
`results/REPORT_opendata.md` for the artifact a tight veto produces).

`opendata/` additionally ships a CMSSW EDAnalyzer (`PFNanoLite`) + one-command
driver (`opendata/run.sh`) that produce the same ntuples inside the official CMS
open-data container — useful as an independent cross-check of the decoder.

## Pipeline

| stage | file | what |
|---|---|---|
| config | `src/config.py` | records, selection cuts, soft-particle definition |
| I/O | `src/io.py` | list/download Open Data over HTTPS |
| MiniAOD decode | `src/miniaod.py` | packedPFCandidates → particles, PF-muon Z selection |
| selection | `src/select.py` | `Z→μμ` selection, targets `y_Z`, `pz`, `β_z` |
| soft particles | `src/softparticles.py` | muon removal, soft/UE selection, PV pileup suppression |
| skim | `src/skim.py` | `make_records` → per-event Parquet (shared schema) |
| simulation | `sim/generate.py` | Pythia8 `Z→μμ` → same Parquet schema |
| dataset | `src/dataset.py` | Parquet → padded tensors, splits, engineered features |
| models | `src/models/efn.py`, `transformer.py` | Energy Flow Network, Particle Transformer |
| train/eval | `src/train.py`, `src/evaluate.py` | Huber multi-target training; metrics, plots |
| baselines | `src/baselines.py` | mean / linear / GBDT (summary + rich engineered) |
| measurement | `src/measure.py` | data-vs-sim matrix with leakage controls |
| ablation | `src/ablation.py` | where the boost info lives: central vs forward |
| W application | `src/wmass.py`, `sim/generate_w.py` | Z-trained model applied to `W→μν` |

Targets: `y_Z` (primary — the Lorentz boost rapidity), `pz`, `β_z`.
Per-particle inputs are longitudinal-info-safe (each particle's own η, log pₜ,
charge, and azimuth *relative to the Z*; never the muon η/pz), and events with
an empty soft set are dropped at skim time.
