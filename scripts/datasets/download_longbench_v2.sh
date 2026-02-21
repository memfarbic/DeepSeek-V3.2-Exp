#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/data/raw/longbench_v2"
OUT_FILE="${OUT_DIR}/data.json"

mkdir -p "${OUT_DIR}"

URL="https://huggingface.co/datasets/THUDM/LongBench-v2/resolve/main/data.json"
echo "Downloading LongBench v2 to ${OUT_FILE}"
curl -L "${URL}" -o "${OUT_FILE}"
echo "Done."

