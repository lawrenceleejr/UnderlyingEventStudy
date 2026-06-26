#!/usr/bin/env bash
# Runs INSIDE the cmsopendata/cmssw_10_6_30 container.
# Builds the PFNanoLite plugin and runs it over the MiniAOD files listed in
# /work/files.txt (root://eospublic... URIs streamed via xrootd), producing one
# flat ntuple per input file in /work/ntuples/.
#
# The repo is mounted at /mnt; the shared work area at /work.
set -e

source /opt/cms/cmsset_default.sh 2>/dev/null || source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /home/cmsusr 2>/dev/null || cd ~

if [ ! -d CMSSW_10_6_30 ]; then
  echo ">>> cmsrel CMSSW_10_6_30"
  cmsrel CMSSW_10_6_30
fi
cd CMSSW_10_6_30/src
eval "$(scramv1 runtime -sh)"   # cmsenv

# install the PFNanoLite plugin from the mounted repo
if [ ! -d PFNanoLite ]; then
  mkdir -p PFNanoLite
  cp -r /mnt/opendata/PFNanoLite/* PFNanoLite/
fi
echo ">>> scram b"
scram b -j"$(nproc)"

mkdir -p /work/ntuples
i=0
while IFS= read -r url; do
  [ -z "$url" ] && continue
  i=$((i+1))
  out="/work/ntuples/pfnanolite_${i}.root"
  if [ -f "$out" ]; then echo "skip existing $out"; continue; fi
  echo ">>> [$i] cmsRun on $url"
  cmsRun /mnt/opendata/pfnanolite_cfg.py \
      inputFiles="$url" outputFile="$out" maxEvents="${MAXEVENTS:--1}"
done < /work/files.txt

echo ">>> produced $(ls /work/ntuples/*.root 2>/dev/null | wc -l) ntuples in /work/ntuples"
