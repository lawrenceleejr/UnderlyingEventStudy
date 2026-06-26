#!/usr/bin/env bash
# Build Pythia8 with Python bindings into /opt/pythia8312.
# Reproducible setup step (binaries are gitignored; rerun after a fresh env).
set -e
VER=pythia8312
SRC=/tmp/${VER}.tgz
PREFIX=/opt/${VER}
if [ -f ${PREFIX}/lib/pythia8.so ]; then echo "already built at ${PREFIX}"; exit 0; fi
[ -f ${SRC} ] || curl -sS -o ${SRC} "https://pythia.org/download/pythia83/${VER}.tgz"
cd /opt && tar xzf ${SRC}
cd ${PREFIX}
PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
./configure --with-python-include=${PYINC} --with-python-bin=$(dirname $(which python3))/
make -j$(nproc)
echo "Pythia built. Add to env:  export PYTHONPATH=${PREFIX}/lib:\$PYTHONPATH"
