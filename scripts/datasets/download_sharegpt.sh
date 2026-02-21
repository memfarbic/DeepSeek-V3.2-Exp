#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/data/raw/sharegpt"
OUT_FILE="${OUT_DIR}/openchat.train.json"

mkdir -p "${OUT_DIR}"

URL="https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset/resolve/main/openchat.train.json"
echo "Downloading ShareGPT (OpenChat snapshot) to ${OUT_FILE}"
curl -L "${URL}" -o "${OUT_FILE}"
echo "Done."

