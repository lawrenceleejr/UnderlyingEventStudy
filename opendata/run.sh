#!/usr/bin/env bash
# One-command CMS Open Data pipeline (run on a machine with Docker, internet,
# and ~100+ GB free disk). It:
#   1. pulls the CMSSW open-data image,
#   2. resolves N MiniAOD file URLs for DoubleMuon Run2016G (record 30505),
#   3. produces flat ntuples of muons + ALL PF candidates (+lostTracks) with the
#      self-contained PFNanoLite EDAnalyzer (no GlobalTag/Frontier needed),
#   4. skims them to Parquet and trains/evaluates the soft-particle -> Z-boost
#      regressor -- the SAME pipeline used for the Pythia study.
#
# Usage:  ./opendata/run.sh [NFILES] [MAXEVENTS_PER_FILE]
set -e

NFILES="${1:-3}"
MAXEVENTS="${2:--1}"
RECORD=30505
IMAGE=cmsopendata/cmssw_10_6_30-slc7_amd64_gcc700
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${REPO}/work"
mkdir -p "${WORK}/ntuples"

echo "== 1. pull CMSSW image =="
docker pull "${IMAGE}"

echo "== 2. resolve ${NFILES} MiniAOD file URLs (record ${RECORD}) =="
python3 - "$RECORD" "$NFILES" > "${WORK}/files.txt" <<'PY'
import sys, requests
rec, n = int(sys.argv[1]), int(sys.argv[2])
md = requests.get(f"https://opendata.cern.ch/api/records/{rec}", timeout=60).json()["metadata"]
uris = [f["uri"] for fi in md.get("_file_indices", []) for f in fi.get("files", [])
        if f.get("uri", "").endswith(".root")]
print("\n".join(uris[:n]))
PY
echo "   $(wc -l < "${WORK}/files.txt") files:"; cat "${WORK}/files.txt"

echo "== 3. produce flat ntuples in CMSSW container =="
docker run --rm -e MAXEVENTS="${MAXEVENTS}" \
    -v "${REPO}:/mnt:ro" -v "${WORK}:/work" \
    "${IMAGE}" bash /mnt/opendata/produce.sh

echo "== 4. skim -> parquet, then train/evaluate =="
cd "${REPO}"
# charged tracks carry real pileup in data: keep PV-associated (use_pv default on)
python3 -m src.skim --local-glob "${WORK}/ntuples/*.root" \
    --out data/skim/opendata.parquet
python3 -m src.evaluate --parquet data/skim/opendata.parquet \
    --models efn transformer --tag opendata

echo "== done. See results/REPORT_opendata.md =="
