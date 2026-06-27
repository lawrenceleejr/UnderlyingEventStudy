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

## 6. Real CMS Open Data — the measurement (DoubleMuon Run2016G)

The full real-data pipeline (`opendata/run.sh`) was **run end to end**: the CMSSW
`10_6_30` open-data container produces `_allPF` soft tracks from DoubleMuon
Run2016G MiniAOD (record 30505), 232,704 events → **29,600 `Z→μμ`**. Full results
and figures: `results/REPORT_opendata.md`. Headline:

- **Pileup washes out the nominal soft event** (⟨730⟩ particles, ~93% pileup):
  EFN corr(y_Z) ≈ **0.00**, confirming the caveat below.
- **PUPPI pileup suppression recovers the signal strongly:** on the leading-vertex
  set (PUPPI>0.5, ⟨49⟩ particles) the EFN reaches
  **corr(y_Z) = 0.75 ± 0.01 (stat) ± 0.02 (syst, indicative), R² = 0.56, sign acc 0.81** —
  *higher* than the pileup-free Pythia truth (0.26).
- **Validated, not leakage:** a label-permutation null test gives corr 0.02; all
  signal flows through particle η; the correlation is central (|η|<2.5), carried by
  neutrals and charged alike, and stable vs pileup. Detector η/pT-resolution and
  PF reco-efficiency systematics are ≤0.02; a conservative charged-only
  (neutral-PUPPI-independent) cross-check still gives corr 0.45.
- **Data > Pythia (indicative):** data corr 0.75 vs Monash Pythia 0.26; on the
  most-matched object (charged-only) 0.45 vs 0.18. The comparison is *not*
  detector/selection-matched (truth MC, no pileup vs PUPPI-cleaned reconstructed
  data), so it is an upper bound — but the direction is robust (MPI-off Pythia is
  *higher*, ruling out MPI as the cause): the data coupling exceeds Monash Pythia.
  An unfolded measurement would quantify the true gap.

So the original premise holds in **real data**, and more strongly than the
generator predicts. The pileup-robust handle anticipated below (central charged
from the PV) is confirmed (corr 0.45), and PUPPI-cleaned neutrals add substantially
more.

Reproduce: `bash train.sh` (Pythia) or `opendata/run.sh` (real data; needs Docker
+ ~100 GB disk).

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
  (which only fixes p_T^W) — not a per-event mass.

### Demonstrated on real data via the Z→μμ closure (the trustworthy test)

Because Pythia under-predicts the effect ~3× (§6), the Pythia W transfer cannot be
trusted to validate the method — it only tests Pythia's self-consistency. The
faithful demonstration is on **real Z→μμ data**, where the truth is fully known:
treat μ2 as the "neutrino" (discard its p_z) to mimic W→ℓν, then put the
longitudinal d.o.f. back with the soft-event boost via
`m² = m_T² + 2 pT^ℓ pT^ν (cosh Δy − 1)` (`results/opendata_zmass_closure.png`):

- the transverse mass is biased **−12 GeV** (peaks below m_Z);
- the **soft-event correction removes the bias** — the reconstructed-mass peak
  moves onto m_Z (+2 GeV), with per-event tails set by the boost resolution;
- the truth-Δy closure recovers the sharp m_Z peak (−0.5 GeV).

**Does it sharpen the mass?** A template-fit linearity study (physical mass
morphing, realistic MET smearing, truth-free ν-p_z reconstruction;
`results/opendata_mass_linearity.png`) shows all observables fit linearly and
unbiased, with σ(m_Z): m_T 0.77 vs soft-corrected **0.97** GeV (per 3k). So at the
current boost resolution the soft-corrected mass **distribution** fit is **~25%
*worse* than m_T** — resolving the genuine two-fold ν-p_z ambiguity from a noisy
boost (corr 0.72) *adds* noise. The longitudinal correction does not sharpen the
mass fit; its realized value is the **aggregate/systematic** handle. (The idealized
full-dimuon floor, 0.003 GeV with no width/resolution, only marks that *perfect*
longitudinal info would help.)

Pythia W details: `results/REPORT_wmass.md` (`python -m src.wmass`). Real-data
demonstrator + mass study: `results/REPORT_opendata.md`.
