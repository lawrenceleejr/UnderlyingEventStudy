# Predicting the Z longitudinal boost from the soft underlying event in CMS Open Data

**Dataset:** CMS Open Data, `DoubleMuon` Run2016G MiniAOD (record
[30505](https://opendata.cern.ch/record/30505)). **Selection:** 232,704 events
processed → **29,600 `Z→μμ`** (opposite-sign μ, pT 20/10 GeV, |η|<2.4, 81<m_μμ<101 GeV).

## Abstract

A permutation-invariant deep set (Energy Flow Network) that sees **only the soft,
non-muon particles** predicts the Z longitudinal boost `y_Z` — measured
independently from the two muons — with

> **corr(y_Z) = 0.75 ± 0.01 (stat) ± 0.02 (syst, indicative),  R² = 0.56,  sign accuracy 0.81**

on held-out data, once pileup is suppressed with PUPPI. The signal is **absent in
the nominal (pileup-dominated) soft event** (corr ≈ 0.00) and is recovered only
after PUPPI pileup removal. It is validated against a label-permutation null test
and is stable against pileup, detector resolution, and reconstruction-efficiency
systematics. The correlation is **markedly larger in data than in Monash Pythia 8**
(0.75 vs 0.26) — though this comparison is *not* detector/selection-matched (truth
MC with no pileup vs reconstructed, PUPPI-cleaned data), so it is an **upper bound**
on generator mismodeling; even the most-matched object (charged-only) gives
data 0.45 vs Pythia 0.18. As a demonstrator for the W mass, adding the soft-event
boost to the dimuon transverse mass removes its bias and pulls the reconstructed
mass onto m_Z (though it does not, at this resolution, sharpen the mass fit).

---

## 1. Method

Schematics: `diagrams/fig_physics.png` (why the soft event encodes the boost),
`diagrams/fig_method.png` (pipeline + EFN), `diagrams/fig_pileup.png` (the pileup
problem).

- **Production.** A self-contained CMSSW `PFNanoLite` EDAnalyzer (no GlobalTag)
  flattens muons + **all** packed PF candidates (+ `lostTracks`); 10 MiniAOD files
  streamed via XRootD inside the `cmssw_10_6_30` open-data container.
- **Soft event.** Non-muon PF candidates, 0.5 < pT < 5 GeV, charged to |η|<2.5
  (PV-associated) and neutrals to |η|<5; muon footprint removed (ΔR, pdgId, pT cap).
- **Pileup suppression.** Keep PUPPI weight > 0.5 (the leading-vertex set).
- **Targets** `y_Z, p_z, β_z` from the dimuon; **per-particle inputs are
  longitudinal-info-safe**: each particle's own η, log pT, charge, PUPPI, dz, and
  azimuth *relative to the Z* — never the muon η/p_z.
- **Model.** Energy Flow Network (shared per-particle Φ → masked sum → event MLP);
  Huber for the point estimate, Gaussian-NLL head for per-event uncertainty.

## 2. Headline result and the pileup story

Figures: `opendata_final_correlation.png`, `opendata_crosschecks.png`,
`opendata_contam_bands.png`.

| selection | ⟨n particles⟩ | EFN corr(y_Z) |
|---|---|---|
| nominal soft event (pileup) | 730 | **0.00** |
| PUPPI > 0.5 (leading vertex) | 49 | **0.75** |

PUPPI weights are **bimodal** — 93% of candidates have weight exactly 0 (pileup) —
so the boost signal is washed out in the full event and recovered as a *step* once
the zero-weight pileup is removed; the result is therefore **insensitive to the
exact PUPPI threshold** (0.3/0.5/0.7 → 0.745/0.751/0.762). The deep set extracts
far more than the best hand-built observable: a linear fit to the pileup-suppressed
UE summary (pT-weighted ⟨η⟩, etc.) reaches only corr 0.25.

## 3. Validation — the result is genuine, not leakage

Figure: `opendata_crosschecks.png`.

- **Label-permutation null test:** shuffling `y_Z` collapses the correlation to
  **0.02** — there is no pipeline/overfitting leakage.
- **All signal flows through η:** in feature-ablation cross-checks, removing the η
  input collapses the correlation to ≈0, and event multiplicity alone carries none
  (corr ≈ 0) — the signal is the particle *rapidity distribution*, not a count or a
  pT effect. The muons are removed, so it is genuinely the *non-muon* soft event
  predicting the muon-derived boost.
- **Where it lives (PUPPI>0.5):** central |η|<2.5 → 0.76, neutral-only → 0.73,
  charged-only → 0.45, forward |η|>2.5 → 0.04. The signal is **central**, not a
  forward/HF artifact, and neutral-only (cannot be muon tracks) carries it strongly.
- **Pileup-stable:** in terciles of pileup the correlation is flat/rising
  (0.71 → 0.76 → 0.78) — not a residual-pileup artifact. Empty-event fraction 0.1%.
- **Architecture cross-check:** a Particle Transformer (a different
  permutation-invariant architecture) agrees with the EFN on sign accuracy
  (0.82 vs 0.81) — the result is not an EFN-specific artifact (the data analog of
  the EFN≈Transformer agreement seen in Pythia, FINDINGS §4).
- **Statistics:** 6 train/test-split+init seeds give corr 0.756 ± 0.008 (std),
  consistent with the test-resampling bootstrap CI [0.743, 0.759].

## 4. Data vs Monte Carlo — the generator under-predicts the effect

The same *nominal cuts* applied to Monash Pythia 8 `Z→μμ`. **The selections are
not physically identical:** Pythia is generator-truth final-state particles with no
pileup (so PUPPI>0.5 is a no-op, ⟨n⟩≈75), while data are detector-reconstructed PF
candidates after PUPPI pileup removal (⟨n⟩≈49). So the comparison is indicative,
not unfolded.

| component | data | Pythia (MC) | ⟨n⟩ data / MC |
|---|---|---|---|
| all (PUPPI>0.5) | **0.75** | 0.26 | 49 / 75 |
| neutral | 0.73 | 0.25 | 16 / 36 |
| central | 0.76 | 0.21 | 47 / 60 |
| **charged (most matched)** | **0.45** | **0.18** | 32 / 39 |

Data is markedly more correlated than Pythia across **every** component, including
the most nearly object-matched (charged-only: 0.45 vs 0.18, ⟨n⟩ 32 vs 39). The
repo's own MPI-off Pythia (corr 0.35 > MPI-on 0.26) rules out the multiparton
interaction as the cause, so the gap is *not* an MPI artifact. **It is an upper
bound on generator mismodeling** — part may be detector/selection/multiplicity
mismatch — but the direction is robust: the data coupling is stronger than Monash
Pythia predicts. Either way the method must be **calibrated on data, not validated
on MC**; an unfolded, detector-matched measurement would quantify the true gap.

## 5. Longitudinal momentum, with calibrated uncertainty

Figure: `opendata_final_residual.png`. The heteroscedastic EFN predicts the
hard-scatter `p_z` with **corr 0.72, R² 0.52**, resolution **107 GeV vs σ_pz = 155 GeV**
(31% tighter than predict-the-mean). The **pull width is 1.07** — the per-event
uncertainties are well calibrated. EFN(MC) barely beats the no-information baseline,
reproducing the data-vs-MC gap.

## 6. W-mass demonstrator: Z→μμ closure

Figure: `opendata_zmass_closure.png`. Treating μ2 as the "neutrino" (longitudinal
momentum discarded) mimics `W→ℓν`, where m_T is the standard observable because
ν p_z is unmeasured. Using the identity

> m_W² = m_T² + 2 pT^ℓ pT^ν (cosh Δy − 1),

the soft event supplies the Δy that m_T discards:

| reconstruction | bias vs m_Z |
|---|---|
| transverse m_T (no longitudinal info) | peaks low, **−12 GeV** |
| m_T + soft-event Δy | **peak → m_Z, +2 GeV** |
| m_T + truth Δy (closure) | −0.5 GeV ✓ |

The soft-event boost **removes the transverse-mass bias** (peak onto m_Z), with
per-event resolution tails set by the boost resolution.

## 7. Does it sharpen a mass measurement?

Figure: `opendata_mass_linearity.png` (template fit, physical mass morphing, with
realistic MET smearing of the neutrino-proxy). All observables fit **linearly and
unbiased** (slope ≈ 1). Per 3000 events:

| observable (distribution template fit) | σ(m_Z) |
|---|---|
| full dimuon (both leptons), *idealized* | 0.003 GeV |
| transverse m_T | 0.77 GeV |
| soft-corrected mass (truth-free ν-p_z) | **0.97 GeV** |

**Honest conclusion.** Fitting the soft-corrected mass *distribution* is **not as
sensitive as m_T** at the current boost resolution — it is ~25% *worse*
(0.97 vs 0.77 GeV). The soft correction must resolve the genuine two-fold ν-p_z
ambiguity from a noisy boost (corr 0.72); resolved without truth (the
constraint-correct root), the ambiguity *adds* noise and the boost's information
does not compensate. The boost carries no mass information by itself — it can only
help by sharpening toward the true peak, which requires a *precise* boost. (An
earlier estimate that put soft ≈ m_T used a truth-oracle branch pick and is
withdrawn.) The full-dimuon entry is an **idealized floor** (the morph gives a
delta with no Z width or detector resolution), not a physical Z-mass uncertainty;
it only marks that *perfect* longitudinal information would help. **The realized
value of the soft-event boost is therefore the aggregate / systematic handle — a
data-driven constraint on the W rapidity / PDF modeling — not a per-event or
distribution-fit statistical gain at this resolution.**

## 8. Systematic uncertainties

Figure: `opendata_systematics.png`.

| source | shift in corr(y_Z) |
|---|---|
| PUPPI threshold 0.3 / 0.7 | −0.006 / +0.011 |
| acceptance \|η\|<2.4 / mass window | −0.003 / +0.003 |
| η resolution 0.03–0.05 / pT resolution 10% | 0.000 / 0.000 |
| PF reconstruction efficiency −2 / −5 / −10% | −0.004 / −0.010 / −0.016 |
| **dominant detector/reco shift** | **≈0.02** (PF eff −10% & PUPPI WP) |

Detector and reconstruction effects are all small (≤0.016; η/pT resolution
negligible — the η-distribution signal is insensitive to them), dominated by the
PF-efficiency and PUPPI working-point variations at ≈0.02. We quote this as an
*indicative* systematic rather than a calibrated quadrature sum (the variations are
single-toy on a shared split). Separately, restricting to **charged-only** tracks
(a different, more conservative object — it discards the PUPPI-weighted neutrals
that carry most of the signal, neutral-only corr 0.73) still gives corr 0.45, far
above Pythia (0.18 charged) and the baselines; it is a robustness *cross-check on
the neutral-PUPPI modeling*, not a smearing systematic.

## 9. Caveats and outlook

- Single run era (2016G); systematics are indicative (Open Data, no full
  calibration chain). The signal lives in the PUPPI-cleaned neutrals (neutral-only
  corr 0.73) and is insensitive to the PUPPI threshold (bimodal weights); the
  neutral-PUPPI *modeling* is the main thing a full measurement must pin down, with
  charged-only (corr 0.45) as the conservative, neutral-independent floor.
- The data-vs-MC discrepancy is the headline physics question: is the stronger
  data correlation a real beam-remnant/ISR feature the generators miss? A dedicated,
  unfolded measurement would settle it.
- The W application is necessarily data-driven (Pythia is unreliable here): train
  the soft-boost estimator on Z data, transfer to W data — the value is an aggregate
  constraint on the W longitudinal/PDF systematic, not a per-event mass.

*Reproduce:* `opendata/run.sh` (production) → `src.skim` → `src.opendata_study`,
`src.opendata_validate`, `src.opendata_systematics`, `src.opendata_pz`, and the
`results/make_*.py` figure scripts.
