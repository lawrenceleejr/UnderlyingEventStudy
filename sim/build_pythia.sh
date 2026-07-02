#!/usr/bin/env bash
# Build Pythia8 with Python bindings.
# Reproducible setup step (binaries are gitignored; rerun after a fresh env).
#
# Honors PYTHIA_PREFIX (default /opt/pythia8312 when writable, else ~/pythia8312)
# so it works without root and on macOS (sysctl fallback for core count).
set -e
VER=pythia8312
SRC="${TMPDIR:-/tmp}/${VER}.tgz"

if [ -n "${PYTHIA_PREFIX}" ]; then
  PREFIX="${PYTHIA_PREFIX}"
elif [ -w /opt ] 2>/dev/null; then
  PREFIX="/opt/${VER}"
else
  PREFIX="${HOME}/${VER}"
fi
PARENT="$(dirname "${PREFIX}")"

if [ -f "${PREFIX}/lib/pythia8.so" ]; then echo "already built at ${PREFIX}"; exit 0; fi
[ -f "${SRC}" ] || curl -sS -o "${SRC}" "https://pythia.org/download/pythia83/${VER}.tgz"
mkdir -p "${PARENT}"
tar xzf "${SRC}" -C "${PARENT}"
# the tarball unpacks as pythia8312; move if a custom prefix name was requested
[ -d "${PREFIX}" ] || mv "${PARENT}/${VER}" "${PREFIX}"
cd "${PREFIX}"
PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
./configure --with-python-include="${PYINC}" --with-python-bin="$(dirname "$(which python3)")/"
NCORES=$( (nproc || sysctl -n hw.ncpu || echo 2) 2>/dev/null )
make -j"${NCORES}"
echo "Pythia built. Add to env:  export PYTHONPATH=${PREFIX}/lib:\$PYTHONPATH"
