# Real CMS Open Data result (DoubleMuon 2016) — and an important artifact

**TL;DR.** The accessible flat open-data sample (jets-only PFNano, record 31305)
cannot deliver a clean soft-underlying-event measurement, for two reasons we
demonstrate below: (1) its PF candidates are *jet constituents*, not the diffuse
soft UE, and (2) a naive muon veto lets the detector's **muon footprint leak the
muon directions** into the "soft" set and fake a strong signal. After removing
that leak, no genuine soft-UE boost signal is seen beyond a modest ~0.20
correlation in simple recoil observables. The clean measurement needs the full
`_allPF` soft tracks + pileup mitigation produced by `opendata/run.sh`.

## Sample
- DoubleMuon Run2016G, jets-only PFNano (record 31305), streamed over HTTPS.
- 63,112 `Z→μμ` events (8 files), Z mass peak at **90.7 GeV** (selection validated).
- Pileup ~20–30 interactions; ⟨n_soft⟩ ≈ 660 candidates/event (jet constituents).

## The muon-footprint artifact (and how we caught it)
With a tight muon veto (ΔR<0.05), the EFN appeared to predict `y_Z` extremely
well — corr **0.53**, sign-accuracy **0.71** — far above truth-level Pythia
(0.26) and the linear baseline on the same data (0.03). That is physically
impossible for a genuine soft-UE signal (real data has pileup + detector smearing
that can only *degrade* it). Diagnostics:

| test | corr(y_Z) | note |
|---|---|---|
| nominal, ΔR<0.05 veto | 0.53 | suspicious |
| **shuffled target** | −0.01 | no code/label leak — pipeline is sound |
| central charged only | 0.10 | weak |
| forward only | 0.01 | ~nothing |
| **muon veto widened to ΔR<0.40** | **−0.02** | **signal gone** |

Widening the muon veto removed only ~11 candidates/event (neutral calo deposits in
the 0.05–0.40 annulus around the muons) yet destroyed the entire signal. The
network had been **reconstructing the muon directions from their detector
footprint** — and the two muon directions fix `y_Z` almost exactly. It was never
the underlying event.

Control: the same widening on **truth-level Pythia** changes nothing
(corr 0.282→0.273), because truth muons leave no detector deposits. So the
simulation result (`FINDINGS.md`) is robust; only the data was contaminated. The
default muon veto is now ΔR<0.4 (`src/config.py`).

## Honest result after removing the leak (ΔR<0.4 veto)

| model | corr(y_Z) | R² | sign acc |
|---|---|---|---|
| predict-the-mean | – | 0.00 | 0.50 |
| linear on recoil-summary obs | **0.20** | 0.04 | 0.58 |
| GBDT on recoil-summary obs | 0.21 | 0.04 | 0.58 |
| Energy Flow Network | 0.01 | 0.00 | 0.50 |

- A genuine but modest **~0.20** correlation survives in the simple η-pₜ-asymmetry
  of the jet constituents. This is the **hard hadronic recoil** (jet fragmentation
  balancing the Z), *not* the diffuse soft underlying event — an artifact of the
  jets-only sample, which only stores particles clustered into jets.
- The raw-particle EFN collapses to the mean here (checked at lr 1e-3/3e-3/1e-2):
  the weak, global η-asymmetry is swamped in ~660-particle events and a sum-pooled
  deep set does not recover what the pre-normalized engineered observable exposes
  directly. (On the clean Pythia soft event the EFN *does* beat the linear
  baseline — the difference is sample quality and signal strength, not the
  architecture.)

## What a real measurement needs
1. **The diffuse soft tracks** — `_allPF` content (all `packedPFCandidates` +
   `lostTracks`), not jet constituents. Produced by `opendata/run.sh` /
   `PFNanoLite` from DoubleMuon MiniAOD (record 30505) on a machine with Docker +
   adequate disk.
2. **Pileup mitigation** — charged tracks from the primary vertex (PUPPI / fromPV);
   the forward neutral region is pileup-dominated in 2016 data.
3. **A wide muon veto** (ΔR≳0.3) or explicit removal of the muons' PF footprint —
   the lesson above.

Reproduce this jets-only check: `python -m src.skim --record 31305 --nfiles 8 --out data/skim/opendata.parquet && python -m src.evaluate --parquet data/skim/opendata.parquet --tag opendata`.
