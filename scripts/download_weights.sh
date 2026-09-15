#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download the FoundationPose ONNX weights from Hugging Face
# (https://huggingface.co/nvidia/foundationpose) and place them under the weights
# directory with the names the rest of this repo expects:
#
#   <WEIGHTS_DIR>/refiner_net.onnx   (RefineNet)
#   <WEIGHTS_DIR>/score_net.onnx     (ScoreNet)
#
# Usage:
#   scripts/download_weights.sh [weights_dir]
#
# The model is public, so no API key is required.
#
# Environment:
#   WEIGHTS_DIR   Target directory (default: ./weights, or $FP_WEIGHTS_DIR if set).
#   FORCE=1       Re-download even if both target files already exist.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_REPO="nvidia/foundationpose"
MODEL_REVISION="18d8309afc9790cddc03a1d50bc69954dc058693"
BASE_URL="https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}"

WEIGHTS_DIR="${1:-${WEIGHTS_DIR:-${FP_WEIGHTS_DIR:-./weights}}}"
REFINER_DST="${WEIGHTS_DIR}/refiner_net.onnx"
SCORE_DST="${WEIGHTS_DIR}/score_net.onnx"

# --- Skip if already present (unless FORCE=1) --------------------------------
if [[ "${FORCE:-0}" != "1" && -f "$REFINER_DST" && -f "$SCORE_DST" ]]; then
  echo "weights already present:"
  echo "  $REFINER_DST"
  echo "  $SCORE_DST"
  echo "(set FORCE=1 to re-download)"
  exit 0
fi

command -v wget >/dev/null 2>&1 || { echo "error: 'wget' is required but not installed" >&2; exit 1; }

mkdir -p "$WEIGHTS_DIR"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# --- Download ----------------------------------------------------------------
echo "=== downloading ${MODEL_REPO} from Hugging Face ==="
for filename in refiner_net.onnx score_net.onnx; do
  url="${BASE_URL}/${filename}"
  echo "    $url"
  if ! wget -O "${WORK_DIR}/${filename}" "$url"; then
    echo "error: failed to download ${filename}. This model is public and needs no key." >&2
    echo "       Verify connectivity to huggingface.co and its download hosts and retry." >&2
    exit 1
  fi
done

# --- Install with the canonical names ----------------------------------------
cp -f "${WORK_DIR}/refiner_net.onnx" "$REFINER_DST"
cp -f "${WORK_DIR}/score_net.onnx" "$SCORE_DST"

echo "=== done ==="
echo "  RefineNet: $REFINER_DST"
echo "  ScoreNet : $SCORE_DST"
ls -la "$REFINER_DST" "$SCORE_DST"
