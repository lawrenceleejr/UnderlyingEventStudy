# Findings: does the soft event predict the Z longitudinal boost?

**Question.** In `Z→μμ`, can a model that sees *only the soft, non-muon particles*
predict the Z's longitudinal boost (rapidity `y_Z`), which is set by the colliding
partons' momentum fractions `x1, x2`?

**Short answer.** Yes — **measured in real CMS collision data**: on 141,965
`Z→μμ` events from DoubleMuon Run2016G open data, a deep set reading only the
soft, PV-associated charged particles predicts `y_Z` with
**corr = 0.142 ± 0.007 (~21σ)**, sign accuracy 0.550, against a shuffled-target
control of 0.005 (`results/REPORT_opendata.md`). At Pythia8 truth level the
matched correlation is 0.200 ± 0.009 — data retains ~70% of the simulated
signal. The predictive power is carried mainly by the **beam-remnant / ISR
fragmentation** kinematically tied to `x1, x2`; the multi-parton-interaction
(MPI) component is largely uncorrelated with the boost and dilutes the signal.

## 0. The real-data measurement (the flagship result)

CMS MiniAOD is decoded directly with uproot (`src/miniaod.py` — no CMSSW), the
Z is built from the PF muons (peak at 90.7 GeV), and the soft set is charged
particles with `0.5 < pT < 5 GeV`, `|η| < 2.5`, leading-PV association, and a
wide ΔR > 0.4 muon veto:

| variant | ⟨n⟩ | linear | GBDT rich | **EFN** | sign acc | shuffle |
|---|---|---|---|---|---|---|
| **DATA · charged-PV** | 28 | 0.046 | 0.130 | **0.142 ± 0.007** | 0.550 | 0.005 |
| SIM · charged-PV | 37 | 0.099 | 0.176 | **0.200 ± 0.009** | 0.571 | −0.001 |
| SIM · full (truth) | 72 | 0.159 | 0.241 | **0.268 ± 0.009** | 0.588 | 0.008 |

In data the simple η-asymmetry nearly vanishes (linear 0.046) while the deep
set extracts 0.142 — the information survives in subtler correlations. The
data/sim ratio (~0.7) is itself physics: Pythia transports more longitudinal
information into the soft charged event than survives in detector data.

## 1. Simulation reference (Pythia8, Monash tune, 13 TeV, 76k `Z→μμ`, veto 0.4)

Soft particles = non-muon final-state particles, `0.5 < pT < 5 GeV`, charged to
`|η|<2.5` and neutrals to `|η|<5`; muons and their footprint removed. Targets
`y_Z`, `pz`, `β_z`. Per-particle inputs are longitudinal-info-safe (each
particle's own η, log pₜ, charge, azimuth *relative to the Z*).

| model | corr(y_Z) | R²(y_Z) | sign acc |
|---|---|---|---|
| predict-the-mean | 0 | 0.00 | 0.50 |
| linear on UE summary obs | 0.16 | 0.03 | 0.55 |
| GBDT on UE summary obs | 0.16 | 0.03 | 0.55 |
| GBDT on rich engineered obs | 0.24 | 0.06 | 0.58 |
| **Energy Flow Network** | **0.27** | **0.068** | **0.59** |

The deep set beats even a rich engineered-feature baseline built from the same
per-particle information (0.27 vs 0.24) — most of the signal is capturable by
well-chosen observables, with a genuine residual advantage for the network.
Sign accuracy 0.59 = it identifies *which* proton's parton was harder 59% of
the time (vs 50% chance).

## 2. Where does the information live? (η / charge ablation)

Same EFN, restricted to particle subsets of the same events:

| subset | corr(y_Z) | sign acc | ⟨n particles⟩ |
|---|---|---|---|
| full (\|η\|<5, all) | 0.26 | 0.59 | 74 |
| central (\|η\|<2.5, all) | 0.22 | 0.57 | 59 |
| central charged (\|η\|<2.5) | 0.19 | 0.56 | 38 |
| forward only (\|η\|>2.5) | 0.16 | 0.55 | 15 |

- The signal is **distributed across the whole soft event**, not concentrated in
  one region; the full event is best.
- Crucially, the **detector-measurable central *charged* underlying event alone**
  (`|η|<2.5`, tracks) reaches corr 0.19 — i.e. the effect should be visible with a
  real tracker, not only with forward calorimetry.
- Forward-only has few particles (⟨15⟩) but is individually informative.

## 3. MPI dilutes the signal (underlying-event ablation)

Re-generating with multi-parton interactions **off** (the UE in the technical
sense is removed; only ISR/FSR + beam-remnant fragmentation remain):

| sample | corr(y_Z) | sign acc | ⟨n particles⟩ |
|---|---|---|---|
| MPI on (nominal) | 0.26 | 0.59 | 74 |
| **MPI off** | **0.35** | **0.61** | 17 |

Turning MPI off *raises* the correlation despite ~4× fewer particles. The boost
information is carried by the fragmentation tied to the primary scatter's
`x1, x2`; the additional MPI particles are uncorrelated with that boost and act as
noise. So the predictive handle is the **beam-remnant/ISR recoil**, more than the
MPI "underlying event" per se.

## 4. Architecture cross-check (EFN vs Particle Transformer)

On a matched held-out subsample, a permutation-invariant Particle Transformer and
the Energy Flow Network agree closely (corr ≈ 0.25 vs 0.23), both well above the
hand-built-observable baselines (≈ 0.16). The signal is in the data, not an
artefact of one architecture. (The Transformer is the heavier model; on a Mac M2
GPU via `train.sh` it trains comfortably — on CPU it is the slow path.)

## 5. Relation to the original hypothesis

The premise — *the soft event encodes the hard-scatter longitudinal boost* — is
now demonstrated in both simulation and collision data (§0). The nuance is which
soft component carries it: not the MPI/UE, but the beam-remnant and
initial-state-radiation fragmentation, exactly the components the
Sjöstrand–Skands beam-remnant model ties to the initiator `x` values. The
data/sim comparison (0.142 vs 0.200) says Pythia somewhat *over*-transports this
correlation into the detector-level soft charged event — the opposite of the
original prior that the generator would not contain it at all, and a handle a
tuned comparison could turn into a beam-remnant/ISR modelling constraint.

## 5b. The muon-footprint lesson (how the measurement had to be protected)

A first pass on the *jets-only* PFNano derived data (record 31305) with a tight
ΔR<0.05 muon veto appeared to work *too well* — EFN corr 0.53, sign-acc 0.71,
above truth-level Pythia. That was a **detector artifact**: neutral calorimeter
deposits from the muons survive near the muon directions, and the network
reconstructs those directions (which fix `y_Z`). The controls:

- shuffled-target control → corr −0.01 (no code/label leak; pipeline sound)
- widen muon veto to ΔR<0.40 → signal gone (−0.02 in the 12k diagnostic, +0.01
  in the full 63k re-skim — both consistent with zero)
- same widening on truth Pythia → corr 0.282→0.273 (**unchanged** — truth muons
  leave no deposits)

Every number in §0 therefore uses the wide ΔR<0.4 veto and a per-variant
shuffle control, and the flagship measurement uses the *diffuse* soft tracks
decoded from MiniAOD, not jet constituents. The lesson stands: **explicit
muon-footprint removal is mandatory in detector data.**

## 6. Caveats

- §1–4 are Pythia8 truth level and serve as the reference for the data
  measurement in §0 (which carries detector effects and pileup for real).
- The correlation magnitude is tune-dependent; the data/sim ratio in §0 is the
  quantity a generator comparison should target.
- **Veto-hole channel:** removing ΔR<0.4 cones around the muons deletes particles
  *as a function of the muon directions*, so in principle a network could locate
  the two depleted cones and infer the muon η (→ y_Z). At truth level the
  0.05→0.40 comparison (corr 0.282→0.273) bounds any hole contribution at the
  percent level, and in data the widened veto *killed* the apparent signal rather
  than creating one — but a dedicated control (e.g. embedding fake cones at random
  η) is the right referee-proof answer and is left as future work.
- **Truncation:** the network reads the `MAX_PARTICLES` highest-pT particles
  (400 by default). Pythia events (⟨n⟩≈72) are never truncated; real-data events
  with neutrals included exceed this (⟨n⟩≈650) and lose their softest tail —
  charged-PV-only data (⟨n⟩≈30) is unaffected.
- **Data muon selection:** real-data Z candidates use PF muons and the mass
  window only (no muon ID / isolation / trigger requirement — pat::Muon flags are
  not in the decoded branches). The on-shell mass window keeps this pure enough
  for a boost regression, but it is not a muon-ID-grade selection.

Reproduce: `bash train.sh` (auto-uses Apple Metal/MPS, CUDA, or CPU).

## 7. The motivating application: W-boson boost for the W mass

The real target is W&rarr;&ell;&nu;, where the neutrino p_z is unmeasured — the
longitudinal degree of freedom that forces W-mass analyses onto the transverse mass
and a PDF-modelled W rapidity distribution (a leading systematic, and central to the
CDF vs LHC tension). A soft-system boost estimator is an *independent, data-driven*
handle on that d.o.f. We train the regressor on Z (boost known from the dilepton)
and apply it to W&rarr;&mu;&nu; (`src/wmass.py`):

- **Z &rarr; W transfer works:** the Z-trained model predicts the W boost with
  corr 0.28 (p_z) / 0.29 (y_W) on W — as well as in-domain on Z (0.27). The
  estimator is **portable**, which is the hard part and the key enabler ("Z
  calibrates W").
- **Per-event constraint is weak (honest):** y_W spread shrinks only 1.49&rarr;1.43
  (~4%); reconstructing m_W event-by-event from the soft-predicted &nu; p_z does
  not beat the no-longitudinal-information case at this resolution.
- **Where the value is:** an *aggregate* data-driven constraint on the W rapidity
  distribution (reducing the PDF/longitudinal systematic), orthogonal to the recoil
  (which only fixes p_T^W) — not a per-event mass. Headroom: heavier models on GPU,
  more statistics, real-data pileup mitigation, and ultimately training on data
  where the beam-remnant/ISR modelling is pinned down.

Details and plots: `results/REPORT_wmass.md`. Run: `python -m src.wmass`.
