#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fetch BEN's submodules and build its optional native engines:
#   extern/ddfast              ddfast, the double-dummy solver (Python extension)
#   extern/BGA/BGADLL.Cpp      the C++ BGADLL (PIMC) on ddfast
# Both are optional: BEN falls back to dds3 and the C# BGADLL when they are not
# built. Idempotent - safe to re-run, e.g. after a pull that moves a submodule.
#
# Usage (from anywhere inside the repo, with BEN's Python environment active):
#     bash build_native.sh
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
PYTHON="${PYTHON:-python}"
failed=0

# 1. Submodules, nested ones included (BGA's BGADLL.Cpp, ddfast's extern/dds).
#    A clone made without --recurse-submodules has none of them yet.
echo "[ben-native] fetching submodules ..."
git submodule sync --recursive
if ! git submodule update --init --recursive; then
    echo "[ben-native] ERROR: git submodule update failed" >&2
    exit 1
fi

# 2. ddfast's Python extension (needs nanobind).
echo "[ben-native] building ddfast ..."
if "$PYTHON" -m pip install -q nanobind && sh extern/ddfast/python/build.sh; then
    :
else
    echo "[ben-native] WARNING: ddfast did not build; BEN will use dds3." >&2
    failed=1
fi

# 3. The C++ BGADLL (PIMC). It opens ddfast at run time, so it builds either way.
echo "[ben-native] building the C++ BGADLL ..."
if ! sh extern/BGA/BGADLL.Cpp/build.sh; then
    echo "[ben-native] WARNING: the C++ BGADLL did not build; PIMC will use the C# BGADLL." >&2
    failed=1
fi

# 4. Report what BEN will use.
echo "[ben-native] preflight ..."
(cd src && "$PYTHON" preflight.py) || failed=1

exit $failed
