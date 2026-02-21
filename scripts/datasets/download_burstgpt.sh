#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/data/raw/burstgpt"
mkdir -p "${OUT_DIR}"

TAG="v1.2"
BASE="https://github.com/HPMLL/BurstGPT/releases/download/${TAG}"

FILES=(
  "BurstGPT_without_fails_1.csv"
)

for f in "${FILES[@]}"; do
  url="${BASE}/${f}"
  out="${OUT_DIR}/${f}"
  echo "Downloading ${url} -> ${out}"
  curl -L "${url}" -o "${out}"
done

echo "Done."

