#!/usr/bin/env bash
# Verify a built wheel before it is published.
#
# This is the release gate: it inspects the wheel's contents, installs that exact
# file into a throw-away virtualenv created outside the checkout, and runs an
# import plus one end-to-end example there. Nothing is rebuilt afterwards -- the
# artifact that passes here is the artifact that gets uploaded.
#
# Usage: build/verify_wheel.sh [DIST_DIR]   (default: dist)
#
# Writes the wheel's SHA-256 to stdout and, when running under GitHub Actions,
# appends a traceability block to $GITHUB_STEP_SUMMARY.

set -euo pipefail

DIST_DIR="${1:-dist}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

shopt -s nullglob
WHEELS=("${DIST_DIR}"/*.whl)
shopt -u nullglob

if [ "${#WHEELS[@]}" -ne 1 ]; then
    echo "FAIL: expected exactly one wheel in ${DIST_DIR}, found ${#WHEELS[@]}" >&2
    if [ "${#WHEELS[@]}" -gt 0 ]; then
        printf '  %s\n' "${WHEELS[@]}" >&2
    fi
    exit 1
fi

WHEEL="$(cd "$(dirname "${WHEELS[0]}")" && pwd)/$(basename "${WHEELS[0]}")"
echo "Verifying wheel: ${WHEEL}"

if command -v sha256sum >/dev/null 2>&1; then
    WHEEL_SHA256="$(sha256sum "${WHEEL}" | cut -d' ' -f1)"
else
    WHEEL_SHA256="$(shasum -a 256 "${WHEEL}" | cut -d' ' -f1)"
fi
echo "sha256: ${WHEEL_SHA256}"

# ---------------------------------------------------------------------------
# 1. Wheel contents
# ---------------------------------------------------------------------------
python - "${WHEEL}" <<'PY'
import sys
import zipfile

wheel = sys.argv[1]
names = zipfile.ZipFile(wheel).namelist()
failures = []

tests = sorted(n for n in names if n.endswith("_test.py"))
if tests:
    failures.append(
        f"the wheel ships {len(tests)} test module(s), e.g. {tests[:3]}; "
        "tests are not part of the public distribution"
    )

for required in ("wax/VERSION", "wax/py.typed"):
    if required not in names:
        failures.append(f"missing required file {required!r}")

top_level = [n for n in names if n.endswith(".dist-info/top_level.txt")]
if len(top_level) != 1:
    failures.append(f"expected exactly one top_level.txt, found {top_level}")
else:
    with zipfile.ZipFile(wheel) as zf:
        packages = zf.read(top_level[0]).decode().split()
    if packages != ["wax"]:
        failures.append(f"top_level.txt should list only 'wax', found {packages}")

if failures:
    for f in failures:
        print(f"FAIL: {f}", file=sys.stderr)
    sys.exit(1)

print(f"OK: wheel contents ({len(names)} files, no tests, VERSION + py.typed present, top_level == ['wax'])")
PY

# ---------------------------------------------------------------------------
# 2. Clean install into a fresh venv outside the checkout
# ---------------------------------------------------------------------------
VENV_PARENT="$(mktemp -d)"
cleanup() { rm -rf "${VENV_PARENT}"; }
trap cleanup EXIT

case "${VENV_PARENT}" in
    "${REPO_ROOT}"*)
        echo "FAIL: temporary venv ${VENV_PARENT} is inside the checkout ${REPO_ROOT}" >&2
        exit 1
        ;;
esac

VENV="${VENV_PARENT}/venv"
echo "Creating clean venv at ${VENV} (outside ${REPO_ROOT})"
python -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --quiet --upgrade pip
"${VENV}/bin/python" -m pip install --quiet "${WHEEL}"

# ---------------------------------------------------------------------------
# 3. Import + end-to-end example, run from outside the checkout so that no
#    source directory can shadow the installed package.
# ---------------------------------------------------------------------------
SMOKE="${VENV_PARENT}/smoke_test.py"
cat > "${SMOKE}" <<'PY'
import os
import pathlib
import sys
import sysconfig

repo_root = pathlib.Path(os.environ["WAX_REPO_ROOT"]).resolve()

import wax

wax_file = pathlib.Path(wax.__file__).resolve()
site_packages = pathlib.Path(sysconfig.get_paths()["purelib"]).resolve()

if not wax_file.is_relative_to(site_packages):
    sys.exit(f"FAIL: wax resolved to {wax_file}, expected it under {site_packages}")
if wax_file.is_relative_to(repo_root):
    sys.exit(f"FAIL: wax resolved to {wax_file}, which is inside the checkout {repo_root}")

expected_version = (repo_root / "src" / "wax" / "VERSION").read_text().strip()
if wax.__version__ != expected_version:
    sys.exit(f"FAIL: installed wax.__version__ == {wax.__version__!r}, expected {expected_version!r}")

print(f"OK: import wax -> {wax_file} (version {wax.__version__})")

import numpy as np
import pandas as pd

from wax.accessors import register_wax_accessors

register_wax_accessors()

index = pd.date_range("2021-01-01", periods=4, freq="D")
series = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
result = series.wax.ewm(alpha=0.5).mean()

expected = np.array([1.0, 1.6666666, 2.4285715, 3.2666667])
np.testing.assert_allclose(np.asarray(result.values, dtype=float), expected, rtol=1e-6)
print(f"OK: end-to-end example Series.wax.ewm(alpha=0.5).mean() -> {list(result.values)}")
PY

echo "Running smoke test from ${VENV_PARENT}"
(
    cd "${VENV_PARENT}"
    WAX_REPO_ROOT="${REPO_ROOT}" "${VENV}/bin/python" "${SMOKE}"
)

echo "PASS: ${WHEEL} is publishable"

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
        echo "### Verified release artifact"
        echo
        echo "| field | value |"
        echo "| --- | --- |"
        echo "| wheel | \`$(basename "${WHEEL}")\` |"
        echo "| sha256 | \`${WHEEL_SHA256}\` |"
        echo "| commit | \`${GITHUB_SHA:-unknown}\` |"
        echo "| ref | \`${GITHUB_REF_NAME:-unknown}\` |"
        echo
        echo "This exact file is what the publish steps upload; nothing is rebuilt after verification."
    } >> "${GITHUB_STEP_SUMMARY}"
fi
