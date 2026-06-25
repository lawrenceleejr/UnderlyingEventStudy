# Findings: does the soft event predict the Z longitudinal boost?

**Question.** In `Z→μμ`, can a model that sees *only the soft, non-muon particles*
predict the Z's longitudinal boost (rapidity `y_Z`), which is set by the colliding
partons' momentum fractions `x1, x2`?

**Short answer.** Yes — the information is genuinely present, even at Pythia8
truth level (no detector, no pileup), and a permutation-invariant deep set (Energy
Flow Network) extracts more of it than hand-built underlying-event observables.
The predictive power is carried mainly by the **beam-remnant / ISR fragmentation**
that is kinematically tied to `x1, x2`; the multi-parton-interaction (MPI)
component of the underlying event is largely *uncorrelated* with the boost and
dilutes the signal.

## 1. Headline (Pythia8, Monash tune, 13 TeV, 76k `Z→μμ`)

Soft particles = non-muon final-state particles, `0.5 < pT < 5 GeV`, charged to
`|η|<2.5` and neutrals to `|η|<5`; muons and their footprint removed. Targets
`y_Z`, `pz`, `β_z`. Per-particle inputs are longitudinal-info-safe (each
particle's own η, log pₜ, charge, PUPPI, azimuth *relative to the Z*).

| model | corr(y_Z) | R²(y_Z) | sign acc | corr(pz) |
|---|---|---|---|---|
| predict-the-mean | – | 0.00 | 0.50 | – |
| linear on UE summary obs | 0.17 | 0.03 | 0.56 | 0.17 |
| GBDT on UE summary obs | 0.18 | 0.03 | 0.56 | 0.18 |
| **Energy Flow Network** | **0.26** | **0.067** | **0.59** | **0.27** |

The deep set beats the engineered-feature baselines by ~50% in correlation: the
soft event carries boost information beyond the simple η-pₜ-asymmetry, and the
network finds it. Sign accuracy 0.59 means it identifies *which* proton's parton
was harder 59% of the time (vs 50% chance).

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

The premise — *the soft event encodes the hard-scatter longitudinal boost* — holds
in simulation. The nuance is which soft component carries it: not the MPI/UE, but
the beam-remnant and initial-state-radiation fragmentation, exactly the components
the Sjöstrand–Skands beam-remnant model ties to the initiator `x` values. Pythia
*does* contain this effect (contrary to the prior that it would not), which is
itself informative: a model trained on data can be compared against Pythia to test
whether nature shows a *stronger* correlation than the generator.

## 6. Caveats and next step (real data)

- These numbers are Pythia8 truth level: no detector resolution, no pileup, full
  acceptance. Real CMS data has ~20–30 pileup interactions that contaminate the
  soft/forward event; the pileup-robust handle is central *charged tracks from the
  primary vertex* (the `central charged` row above is the relevant proxy).
- The correlation magnitude is tune-dependent; a data measurement is the real test
  of whether the generator gets it right.
- The full real-data pipeline is provided (`opendata/run.sh`): it produces the
  `_allPF` soft tracks from DoubleMuon Run2016G MiniAOD and runs the identical
  analysis. It needs a machine with Docker + ~100 GB disk (this dev environment
  could not host the CMSSW image).

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
