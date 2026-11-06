#!/bin/sh
# Run LlamaF on the ZCU102. On the board, after Linux boots:
#   source /mnt/sd-mmcblk0p1/run_hw.sh ["prompt"] [steps]
#
# Needs these files on the SD card's boot partition:
#   app.exe, vadd.xclbin, xrt.ini, tok_tl-chat.bin, tinyllama-1.1b-chat-v0.2-w8a8-gs256.bin

mount /dev/mmcblk0p1 /mnt 2>/dev/null
cd /mnt

PROMPT="${1:-Once upon a time}"
STEPS="${2:-256}"

# -t 0 = greedy decoding (deterministic output); prints "achieved tok/s" at the end
./app.exe tinyllama-1.1b-chat-v0.2-w8a8-gs256.bin -z tok_tl-chat.bin -t 0 -n "$STEPS" -i "$PROMPT"
