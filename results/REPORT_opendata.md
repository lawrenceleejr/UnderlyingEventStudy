# The real-data measurement: the soft underlying event predicts the Z boost in CMS collision data

**Headline.** In 141,965 `Z→μμ` events from CMS DoubleMuon Run2016G open data,
a deep-set regressor trained on **only the soft, PV-associated charged particles**
(⟨28⟩ per event, muons and their footprint removed) predicts the Z rapidity with

> **corr(pred, true y_Z) = 0.142 ± 0.007** (≈21σ from zero), sign accuracy 55.0%

against a shuffled-target control of 0.005. To our knowledge this is the first
demonstration in real collision data that the soft event carries measurable
information about the hard scatter's longitudinal boost. The matched Pythia8
prediction is 0.200 ± 0.009 — data retains about 70% of the simulated
correlation, consistent with detector resolution, PF/vertex-association
impurity, and residual pileup.

## How the data was read (no CMSSW)

The diffuse soft tracks exist only in MiniAOD's `packedPFCandidates`.
`src/miniaod.py` decodes CMS's packed format **directly with uproot** — IEEE
half-precision minifloats for pT/m/dz, scaled int16 for η/φ, PV-association
quality bits, and the associated-vertex key — turning full MiniAOD collision
data into analysis arrays with no CMSSW, no Docker. Validation: symmetric η/φ
distributions, dz(PV tracks) peaked at 0, Z peak at **90.7 GeV** from PF muons
(13.5% selection efficiency, matching the PFNano-based selection).

Selection: two OS PF muons (pT > 20/10 GeV, |η| < 2.4), 81 < m_μμ < 101 GeV.
Soft set: 0.5 < pT < 5 GeV, ΔR > 0.4 from either muon; *charged-PV* variant
requires association to the leading PV with quality ≥ CompatibilityDz.

## The measurement matrix

| variant | ⟨n⟩ | linear | GBDT rich | **EFN** | sign acc | shuffle ctrl |
|---|---|---|---|---|---|---|
| **DATA · charged-PV** | 28 | 0.046 | 0.130 | **0.142 ± 0.007** | 0.550 | 0.005 |
| DATA · full (+neutrals) | 468 | 0.093 | 0.104 | 0.003 | 0.497 | −0.001 |
| SIM · charged-PV (Pythia8) | 37 | 0.099 | 0.176 | **0.200 ± 0.009** | 0.571 | −0.001 |
| SIM · full (truth, no PU) | 72 | 0.159 | 0.241 | **0.268 ± 0.009** | 0.588 | 0.008 |

(`results/metrics_measurement.json`; errors are Fisher standard errors; the
DATA·full row uses a 70k-event subsample for memory reasons.)

Reading the table:
- **The signal is real in data** and needs more than a simple η-asymmetry: the
  linear baseline nearly vanishes in data (0.046) while the deep set and the
  rich engineered baseline extract 0.13–0.14 — in data the information sits in
  subtler correlations than in truth-level simulation.
- **Data/sim ratio ≈ 0.7** in the matched charged-PV variant. Pythia transports
  more longitudinal information into the soft charged event than survives in
  detector data — a statement a tuned generator comparison could sharpen into a
  constraint on beam-remnant/ISR modelling.
- **Pileup burns the signal**: adding the vertex-less neutrals (~440
  pileup-dominated particles on top of 28 signal tracks) collapses the deep set
  to zero (0.003) while engineered observables keep a residual 0.10 — under
  heavy pileup the deep set cannot isolate the informative minority, and
  PV-associated charged tracks are the only clean carrier. In truth-level sim
  (no pileup) the same neutrals *help* (0.200→0.268).

## The artifact this measurement had to survive (integrity control)

An earlier pass on the *jets-only* PFNano derived sample (record 31305) with a
tight ΔR < 0.05 muon veto produced corr = 0.53 — spectacular and **wrong**:

| test | corr(y_Z) |
|---|---|
| tight veto (ΔR<0.05) | 0.53 |
| shuffled target | −0.01 (pipeline clean) |
| **veto widened to ΔR<0.40** | **≈ 0 (signal gone)** |
| same widening, truth Pythia | 0.282→0.273 (unchanged) |

The network had been reading the muons' own calorimeter deposits near the muon
directions (which fix y_Z), not the underlying event. Every number in the table
above therefore uses the wide ΔR < 0.4 veto, and the shuffled-target control is
run per variant. The jets-only sample itself is unusable for this measurement —
its PF candidates are jet constituents, not the diffuse soft event.

## Caveats
- PF-muon Z selection (no muon-ID flags in the decoded branches); the on-shell
  mass window supplies purity.
- PUPPI weights are not decoded (nonlinear 8-bit packing); charged-PV
  association carries the pileup suppression, neutrals get none.
- corr = 0.142 means the soft event explains ~2% of the y_Z variance per event —
  this is an existence measurement and a generator-modelling probe, not a
  per-event kinematic constraint (see REPORT_wmass for the aggregate use case).
- Veto-hole information channel bounded small at truth level (0.282→0.273) but
  not yet excluded with a dedicated embedding control.

## Reproduce

```bash
python -m src.skim --miniaod --record 30505 --nfiles 20 --out data/skim/miniaod.parquet
python -m src.measure --data data/skim/miniaod.parquet --sim data/skim/pythia.parquet
```
