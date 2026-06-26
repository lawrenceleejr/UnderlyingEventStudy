"""Soft underlying-event particle selection and per-particle feature building.

Given the Z->mumu selected events, build the collection of *soft* particles that
the network is allowed to see: the two muons (and their footprint) are removed,
the hard recoil is suppressed with a pT cap, and pileup is suppressed with the
primary-vertex association quality / PUPPI weight.

The per-particle feature vector is deliberately free of any direct longitudinal
information about the muons: we pass each particle's own (eta, pt, ...) and its
azimuth *relative to the Z* (phi is boost-neutral), never the muon eta/pz.
"""
from __future__ import annotations

import awkward as ak
import numpy as np
import vector

from . import config

vector.register_awkward()

# All PF-candidate branches we read. pvAssocQuality is the MiniAOD "fromPV"-like
# flag (0..7, higher = better PV association); puppiWeight downweights pileup.
PF_BRANCHES = [
    "nPFCands",
    "PFCands_pt",
    "PFCands_eta",
    "PFCands_phi",
    "PFCands_mass",
    "PFCands_charge",
    "PFCands_pdgId",
    "PFCands_puppiWeight",
    "PFCands_dz",
    "PFCands_pvAssocQuality",
]


def pdgid_to_cat(abs_pdg: ak.Array) -> ak.Array:
    """Map |pdgId| to a small category index used for an embedding.

    0 charged hadron, 1 electron, 2 muon, 3 photon, 4 neutral hadron, 5 other.
    """
    cat = ak.zeros_like(abs_pdg)
    cat = ak.where(abs_pdg == 11, 1, cat)
    cat = ak.where(abs_pdg == 13, 2, cat)
    cat = ak.where(abs_pdg == 22, 3, cat)
    cat = ak.where(abs_pdg == 130, 4, cat)
    cat = ak.where(
        (abs_pdg != 211) & (abs_pdg != 11) & (abs_pdg != 13)
        & (abs_pdg != 22) & (abs_pdg != 130),
        5,
        cat,
    )
    return cat


def build_pfcands(arr: ak.Array) -> ak.Array:
    return ak.zip(
        {
            "pt": arr.PFCands_pt,
            "eta": arr.PFCands_eta,
            "phi": arr.PFCands_phi,
            "mass": arr.PFCands_mass,
            "charge": arr.PFCands_charge,
            "pdgId": arr.PFCands_pdgId,
            "puppi": arr.PFCands_puppiWeight,
            "dz": arr.PFCands_dz,
            "pvq": arr.PFCands_pvAssocQuality,
        },
        with_name="Momentum4D",
    )


def select_soft(
    pf: ak.Array,
    m1: ak.Array,
    m2: ak.Array,
    z: ak.Array,
    *,
    include_neutral: bool = True,
    use_pv: bool = True,
    pt_cap: float = config.SOFT_PT_CAP,
) -> ak.Array:
    """Return per-event jagged soft-particle records with network features.

    ``pf`` must already be restricted to the selected events (same length/order
    as ``m1``/``m2``/``z``).
    """
    charged = pf.charge != 0

    # --- muon footprint removal (delta-R to either selected muon) ---
    dr1 = pf.deltaR(m1)
    dr2 = pf.deltaR(m2)
    not_muon_fp = (dr1 > config.MUON_VETO_DR) & (dr2 > config.MUON_VETO_DR)
    not_muon_id = abs(pf.pdgId) != 13

    # --- soft / acceptance selection ---
    soft_pt = (pf.pt > config.SOFT_PT_MIN) & (pf.pt < pt_cap)
    in_trk = abs(pf.eta) < config.TRK_ETA_MAX
    in_fwd = abs(pf.eta) < config.NEUTRAL_ETA_MAX

    keep_charged = charged & soft_pt & in_trk
    if use_pv:
        # pvAssocQuality >= 5 (UsedInFitTight/Loose) suppresses pileup tracks
        keep_charged = keep_charged & (pf.pvq >= 5)

    if include_neutral:
        keep_neutral = (~charged) & soft_pt & in_fwd
        keep = (keep_charged | keep_neutral) & not_muon_fp & not_muon_id
    else:
        keep = keep_charged & not_muon_fp & not_muon_id

    s = pf[keep]

    # --- per-particle features (longitudinal-info-safe) ---
    dphi = (s.phi - z.phi + np.pi) % (2 * np.pi) - np.pi  # azimuth relative to Z
    feats = ak.zip(
        {
            "log_pt": np.log(s.pt),
            "eta": s.eta,
            "dphi": dphi,
            "sin_dphi": np.sin(dphi),
            "cos_dphi": np.cos(dphi),
            "charge": s.charge,
            "puppi": s.puppi,
            "dz": np.minimum(np.maximum(s.dz, -20), 20),
            "cat": pdgid_to_cat(abs(s.pdgId)),
        }
    )
    return feats


def event_summary(soft: ak.Array) -> dict:
    """Cheap event-level scalars used for sanity checks and analytic baselines."""
    n = ak.num(soft)
    pt = np.exp(soft.log_pt)
    sum_pt = ak.sum(pt, axis=1)
    # eta-weighted asymmetry: the leading-order signal for the boost sign
    sum_pt_eta = ak.sum(pt * soft.eta, axis=1)
    eta_ptw = ak.where(sum_pt > 0, sum_pt_eta / sum_pt, 0.0)
    nfwd = ak.sum(soft.eta > 0, axis=1)
    nbwd = ak.sum(soft.eta < 0, axis=1)
    fb_asym = ak.where((nfwd + nbwd) > 0, (nfwd - nbwd) / (nfwd + nbwd), 0.0)
    return {
        "n_soft": np.asarray(n),
        "sum_pt": np.asarray(sum_pt),
        "eta_ptweighted": np.asarray(eta_ptw),
        "fb_asym": np.asarray(fb_asym),
    }
