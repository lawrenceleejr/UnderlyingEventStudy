"""Z -> mu mu selection and longitudinal-boost target computation."""
from __future__ import annotations

import awkward as ak
import numpy as np
import vector

from . import config

vector.register_awkward()

MUON_BRANCHES = ["nMuon", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge"]


def build_muons(arr: ak.Array) -> ak.Array:
    """Pt-sorted Momentum4D muon collection from NanoAOD branches."""
    mu = ak.zip(
        {
            "pt": arr.Muon_pt,
            "eta": arr.Muon_eta,
            "phi": arr.Muon_phi,
            "mass": arr.Muon_mass,
            "charge": arr.Muon_charge,
        },
        with_name="Momentum4D",
    )
    return mu[ak.argsort(mu.pt, ascending=False)]


def zmumu_mask_and_z(arr: ak.Array):
    """Return (event_mask, leadmu, submu, Z) for events passing the Z->mumu selection.

    Selection: >=2 muons, opposite sign on the two leading muons, pT(20/10),
    |eta|<2.4, and dimuon mass in [81,101] GeV. ``leadmu``/``submu``/``Z`` are
    defined only on the selected events (already masked).
    """
    has2 = arr.nMuon >= 2
    mu = build_muons(arr)
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
    full_mask = np.zeros(len(arr), dtype=bool)
    full_mask[np.where(np.asarray(has2))[0]] = np.asarray(good)
    return full_mask, m1[good], m2[good], z[good]


def targets(z: ak.Array) -> dict:
    """Longitudinal-boost targets from the Z 4-vector.

    y_Z   = 0.5 ln((E+pz)/(E-pz))  -- the Lorentz boost rapidity of the Z (primary)
    pz_Z  = longitudinal momentum [GeV]
    beta_z= pz/E
    """
    E = np.asarray(z.energy)
    pz = np.asarray(z.pz)
    return {
        "y_Z": 0.5 * np.log((E + pz) / (E - pz)),
        "pz_Z": pz,
        "beta_z": pz / E,
        "mass": np.asarray(z.mass),
        "pt_Z": np.asarray(z.pt),
        "phi_Z": np.asarray(z.phi),
    }
