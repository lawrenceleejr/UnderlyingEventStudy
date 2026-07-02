#!/usr/bin/env bash
# One-line training launcher. Auto-detects the fastest backend:
#   Apple Metal (MPS) on an M-series Mac, CUDA on NVIDIA, else CPU.
#
# Usage (from the repo root):
#     bash train.sh
# Optional: bash train.sh "efn transformer" 60      # models, epochs
set -e

MODELS="${1:-efn transformer}"
EPOCHS="${2:-50}"

# prefer the full local sample, else the committed compact demo (35k events)
if [ -f "data/skim/pythia.parquet" ]; then
  DATA="data/skim/pythia.parquet"
else
  DATA="data/skim/pythia_demo.parquet"
fi

python3 -m pip install -q -r requirements.txt

if [ ! -f "${DATA}" ]; then
  echo ">> No dataset found; generating one with Pythia8 (one-time)."
  DATA="data/skim/pythia.parquet"
  # pick a writable prefix and hand it to the build script
  PYDIR="${PYTHIA_PREFIX:-$([ -w /opt ] 2>/dev/null && echo /opt/pythia8312 || echo "${HOME}/pythia8312")}"
  export PYTHIA_PREFIX="${PYDIR}"
  bash sim/build_pythia.sh
  export PYTHONPATH="${PYDIR}/lib:${PWD}"
  python3 -m sim.generate --nevents 200000 --out "${DATA}"
fi

echo ">> Training on the best available device (mps/cuda/cpu auto-detected)."
PYTHONPATH="${PWD}" python3 -m src.evaluate \
    --parquet "${DATA}" --models ${MODELS} --epochs "${EPOCHS}" --tag pythia

echo ">> Done. See results/summary_pythia.md and results/*.png"
