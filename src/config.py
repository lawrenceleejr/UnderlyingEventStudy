"""Central configuration for the Z-boost-from-soft-particles analysis.

The whole pipeline reads CMS Open Data ROOT files over HTTPS through the
opendata.cern.ch front-end (which carries a publicly trusted certificate and
supports HTTP range requests, so uproot can stream them without xrootd).
"""
from __future__ import annotations

import os
import ssl
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Remote access to CMS Open Data
# ----------------------------------------------------------------------------
# In this environment outbound HTTPS is intercepted by an egress gateway whose
# CA lives in the bundle below. The gateway trusts opendata.cern.ch (public CA)
# but NOT eospublic.cern.ch (CERN Grid CA) and blocks xrootd port 1094, so we
# always rewrite root://eospublic.cern.ch//eos/... -> https://opendata.cern.ch/eos/...
CA_BUNDLE = os.environ.get("CCR_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
OPENDATA_HTTPS = "https://opendata.cern.ch/eos/"


def ssl_context() -> ssl.SSLContext:
    """SSL context trusting the egress-gateway CA (falls back to system trust)."""
    if os.path.exists(CA_BUNDLE):
        return ssl.create_default_context(cafile=CA_BUNDLE)
    return ssl.create_default_context()


def to_https(uri: str) -> str:
    """Rewrite a CMS Open Data file URI to the reachable HTTPS front-end."""
    if uri.startswith("http"):
        return uri
    if "//eos/" in uri:  # root://eospublic.cern.ch//eos/opendata/...
        return OPENDATA_HTTPS + uri.split("//eos/", 1)[1]
    if uri.startswith("/eos/"):
        return OPENDATA_HTTPS + uri[len("/eos/"):]
    return uri


# ----------------------------------------------------------------------------
# Datasets (CERN Open Data record numbers)
# ----------------------------------------------------------------------------
# Dev fixture: DoubleMuon Run2016G NanoAOD-PF (record 31305). PFCands here are
# JET-CONSTITUENTS ONLY -> good for plumbing, NOT a true underlying-event sample.
RECORD_PFNANO_JETSONLY = 31305
# Real soft tracks come from DoubleMuon Run2016G MiniAOD (record 30505) after a
# one-time CMSSW "_allPF" production (see docker/).
RECORD_MINIAOD = 30505

# ----------------------------------------------------------------------------
# Z -> mumu selection
# ----------------------------------------------------------------------------
MU_PT_LEAD = 20.0
MU_PT_SUBLEAD = 10.0
MU_ETA_MAX = 2.4
Z_MASS_LO = 81.0
Z_MASS_HI = 101.0

# ----------------------------------------------------------------------------
# Soft-particle (underlying-event) definition
# ----------------------------------------------------------------------------
SOFT_PT_MIN = 0.5          # charged-track floor in packed PF candidates
SOFT_PT_CAP = 5.0          # "soft": below this; >cap is hard-recoil, studied via ablation
TRK_ETA_MAX = 2.5          # tracker acceptance for charged candidates
NEUTRAL_ETA_MAX = 5.0      # neutrals extend to HF (forward beam-x proxy)
MUON_VETO_DR = 0.4         # remove the muon footprint; MUST be wide for real data:
# in detector data the muon leaves neutral calo deposits out to dR~0.3 that otherwise
# let a network reconstruct the muon directions (=> y_Z) and fake a "signal".
# Truth-level Pythia is insensitive to this (0.05 and 0.40 agree), but data is not.
MAX_PARTICLES = 400        # padding cap per event for the network

# ----------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------
TARGETS = ["y_Z", "pz_Z", "beta_z"]
PRIMARY_TARGET = "y_Z"

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
SEED = 1234
SPLIT = (0.70, 0.15, 0.15)  # train / val / test
