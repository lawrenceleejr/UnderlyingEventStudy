"""Direct uproot decoding of CMS MiniAOD packedPFCandidates — no CMSSW needed.

MiniAOD stores every PF candidate in ``pat::PackedCandidate`` with lossy packing
(DataFormats/PatCandidates/src/PackedCandidate.cc):

- ``packedPt_``, ``packedM_``, ``packedDz_``/``packedDxy_`` : IEEE-754 half
  precision (CMS ``MiniFloatConverter`` == numpy ``float16``); dz/dxy are
  additionally scaled by 100 (stored in units of 10 um).
- ``packedEta_`` : int16, eta = i16 * 6.0  / 32767
- ``packedPhi_`` : int16, phi = i16 * 3.2  / 32767
- ``pdgId_``     : the PF particle id (+-211 h, +-11 e, +-13 mu, 22 gamma,
  130 K0L, 1/2 HF hadron/em); charge follows from the id.
- ``qualityFlags_`` bits 0-2 : primary-vertex association quality
  (7 UsedInFitTight, 6 UsedInFitLoose, 5 CompatibilityDz, ...).
- ``pvRefKey_``  : index of the associated vertex (0 = leading PV).

These branches are fully split in the ROOT file, so uproot reads them as plain
jagged arrays. This module decodes them into the same record layout the rest of
the pipeline consumes (see ``softparticles.build_pfcands``), which means the
identical selection / skim / model code runs on real MiniAOD collision data.

The muons come from the PF candidates themselves (pdgId +-13): for an on-shell
Z->mumu selection the dimuon mass window provides the purity that a pat::Muon ID
would otherwise supply (validated: Z peak at 90.6 GeV, ~13.5% selection
efficiency on DoubleMuon 2016G, matching the PFNano-based selection).

Limitations (documented, not hidden):
- ``puppiWeight`` uses a nonlinear 8-bit log packing; we do not decode it and set
  puppi = 1.0. Pileup suppression instead uses the PV association (charged
  candidates must have pvRefKey==0 and quality >= CompatibilityDz), which is the
  standard CHS-style selection.
- pat::Muon (slimmedMuons) is not split in the file, so muon ID flags are not
  available; the mass window substitutes.
"""
from __future__ import annotations

import awkward as ak
import numpy as np
import uproot
import vector

from . import config, select

vector.register_awkward()

_PF = ("patPackedCandidates_packedPFCandidates__PAT./"
       "patPackedCandidates_packedPFCandidates__PAT.obj/"
       "patPackedCandidates_packedPFCandidates__PAT.obj.")

_FIELDS = ["packedPt_", "packedEta_", "packedPhi_", "packedM_",
           "pdgId_", "qualityFlags_", "pvRefKey_", "packedDz_"]

BRANCHES = [_PF + f for f in _FIELDS]

MU_MASS = 0.1056584


def _half(u16: ak.Array) -> ak.Array:
    """uint16 minifloat -> float32 (CMS MiniFloatConverter is IEEE half)."""
    flat = np.asarray(ak.flatten(u16)).astype(np.uint16).view(np.float16).astype(np.float32)
    return ak.unflatten(flat, ak.num(u16))


def _i16(u16: ak.Array, scale: float) -> ak.Array:
    """uint16 -> signed int16 -> float32 * scale (eta/phi packing)."""
    flat = np.asarray(ak.flatten(u16)).astype(np.uint16).view(np.int16).astype(np.float32) * scale
    return ak.unflatten(flat, ak.num(u16))


def _charge_from_pdg(pdg: ak.Array) -> ak.Array:
    """PF pdgId -> charge. h+-(+-211) carry sign(pdg); e/mu (11/13) the opposite."""
    apdg = abs(pdg)
    sign = ak.where(pdg > 0, 1.0, -1.0)
    q = ak.where(apdg == 211, sign, 0.0)
    q = ak.where((apdg == 11) | (apdg == 13), -sign, q)
    return q


def decode_pfcands(raw: ak.Array) -> ak.Array:
    """Decode a batch of packed-candidate branches into the pipeline's pf record.

    The returned record matches ``softparticles.build_pfcands`` output:
    Momentum4D with pt/eta/phi/mass + charge, pdgId, puppi, dz, pvq.
    ``pvq`` is zeroed for charged candidates not associated to the leading PV
    (pvRefKey != 0), so the downstream cut ``pvq >= 5`` implements the standard
    fromPV-style pileup rejection with no changes to ``select_soft``.
    """
    pt = _half(raw[_PF + "packedPt_"])
    eta = _i16(raw[_PF + "packedEta_"], 6.0 / 32767.0)
    phi = _i16(raw[_PF + "packedPhi_"], 3.2 / 32767.0)
    mass = _half(raw[_PF + "packedM_"])
    pdg = raw[_PF + "pdgId_"]
    charge = _charge_from_pdg(pdg)
    dz = _half(raw[_PF + "packedDz_"]) / 100.0
    pvq_raw = raw[_PF + "qualityFlags_"] & 7
    pvkey = raw[_PF + "pvRefKey_"]
    is_charged = charge != 0
    pvq = ak.where(is_charged & (pvkey != 0), 0, pvq_raw)
    return ak.zip(
        {
            "pt": pt, "eta": eta, "phi": phi, "mass": mass,
            "charge": charge,
            "pdgId": pdg,
            "puppi": ak.ones_like(pt),   # puppi packing not decoded; PV cut does the work
            "dz": dz,
            "pvq": pvq,
        },
        with_name="Momentum4D",
    )


def zmumu_from_pf(pf: ak.Array):
    """Z->mumu selection using PF muons (pdgId +-13) from the packed candidates.

    Mirrors ``select.zmumu_mask_and_z`` (same cuts from config); returns
    (event_mask, leadmu, submu, Z) with the latter three already masked.
    """
    is_mu = abs(pf.pdgId) == 13
    mu = pf[is_mu]
    mu = ak.zip(
        {"pt": mu.pt, "eta": mu.eta, "phi": mu.phi,
         "mass": ak.ones_like(mu.pt) * MU_MASS, "charge": mu.charge},
        with_name="Momentum4D",
    )
    mu = mu[ak.argsort(mu.pt, ascending=False)]
    has2 = ak.num(mu) >= 2
    mu2 = mu[has2]
    m1, m2 = mu2[:, 0], mu2[:, 1]
    z = m1 + m2
    good = (
        (m1.charge != m2.charge)
        & (m1.pt > config.MU_PT_LEAD)
        & (m2.pt > config.MU_PT_SUBLEAD)
        & (abs(m1.eta) < config.MU_ETA_MAX)
        & (abs(m2.eta) < config.MU_ETA_MAX)
        & (z.mass > config.Z_MASS_LO)
        & (z.mass < config.Z_MASS_HI)
    )
    full_mask = np.zeros(len(pf), dtype=bool)
    full_mask[np.where(np.asarray(has2))[0]] = np.asarray(good)
    return full_mask, m1[good], m2[good], z[good]


def process_file(path: str, step_size: int = 20000, **soft_kw) -> ak.Array:
    """Skim one MiniAOD file -> the standard per-event record array (or None)."""
    from . import skim  # deferred: skim imports this module's sibling helpers

    pieces = []
    for raw in uproot.iterate({path: "Events"}, BRANCHES, step_size=step_size):
        pf = decode_pfcands(raw)
        mask, m1, m2, z = zmumu_from_pf(pf)
        if mask.sum() == 0:
            continue
        rec = skim.make_records(m1, m2, z, pf[mask], **soft_kw)
        if rec is not None and len(rec):
            pieces.append(rec)
    if not pieces:
        return None
    return ak.concatenate(pieces)
