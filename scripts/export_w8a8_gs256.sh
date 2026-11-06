#!/usr/bin/env bash
# Generate the group-wise W8A8 (GS = 256) TinyLlama checkpoint that LlamaF's host
# program (host/run.cpp) and HLS kernels (kernels/matmul{1,2}.cpp) expect.
#
#   Usage: scripts/export_w8a8_gs256.sh [output.bin] [hf_model_dir]
#
# Defaults:
#   output.bin   = models/tinyllama-1.1b-chat-v0.2-w8a8-gs256.bin
#   hf_model_dir = models/TinyLlama-1.1B-Chat-v0.2   (downloaded if missing)
#
# Source model (FP32, 4.4 GB), pinned to the revision used in the paper:
#   https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.2
#   revision c907b2fa3103c3646ca02870ee4b91ce4e260c31 (2023-09-17)
#
# Output format: llama2.c "version 2" checkpoint ("Q8_0" in llama2.c), GS = 256,
# int8 weights and fp32 per-group scales interleaved per tensor, separate
# (unshared) classifier. Expected size: 1,117,524,416 bytes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/models/tinyllama-1.1b-chat-v0.2-w8a8-gs256.bin}"
HF_DIR="${2:-$ROOT/models/TinyLlama-1.1B-Chat-v0.2}"
HF_REPO="TinyLlama/TinyLlama-1.1B-Chat-v0.2"
HF_URL="https://huggingface.co/$HF_REPO"
HF_REVISION="${HF_REVISION:-c907b2fa3103c3646ca02870ee4b91ce4e260c31}"

mkdir -p "$(dirname "$OUT")"

# 1. Download the original HuggingFace model if it is not there yet.
#    Only the safetensors weights, config and tokenizer files (skips the duplicate pytorch_model.bin).
if [ ! -f "$HF_DIR/config.json" ]; then
    echo "[1/3] Downloading $HF_URL (revision ${HF_REVISION:0:8}) -> $HF_DIR"
    python3 - "$HF_REPO" "$HF_REVISION" "$HF_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, rev, out = sys.argv[1:4]
snapshot_download(repo_id=repo, revision=rev, local_dir=out,
                  allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])
EOF
else
    echo "[1/3] Using existing HF model at $HF_DIR"
fi

# 2. Quantize to W8A8 with GS = 256 (llama2.c version 2 export).
echo "[2/3] Exporting W8A8 (GS = 256) checkpoint -> $OUT"
python3 "$ROOT/export/export.py" "$OUT" --version 2 --hf "$HF_DIR"

# 3. Check the header (group size, model shape) and file size.
echo "[3/3] Verifying $OUT"
python3 "$ROOT/scripts/verify_bin.py" "$OUT"
