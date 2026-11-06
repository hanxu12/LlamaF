#!/usr/bin/env python3
"""Check that a checkpoint matches what LlamaF's host program and kernels expect.

Usage: python3 scripts/verify_bin.py models/tinyllama-1.1b-chat-v0.2-w8a8-gs256.bin [--sha256]

Uses only the Python standard library.
"""
import hashlib
import os
import struct
import sys

EXPECTED = dict(dim=2048, hidden_dim=5632, n_layers=22, n_heads=32,
                n_kv_heads=4, vocab_size=32003, group_size=256)
EXPECTED_SIZE = 1_117_524_416
# sha256 of the checkpoint used for the paper's experiments
REFERENCE_SHA256 = "a8df6c39606932456c566e67ef75a0622a9187d83dfcbbad5394bdd31194c111"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    with open(path, "rb") as f:
        hdr = f.read(256)
        magic, version = struct.unpack("Ii", hdr[:8])
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = \
            struct.unpack("7i", hdr[8:36])
        shared_classifier = hdr[36]
        group_size = struct.unpack("i", hdr[37:41])[0]

        got = dict(dim=dim, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
                   n_kv_heads=n_kv_heads, vocab_size=vocab_size, group_size=group_size)
        ok = True
        if magic != 0x616B3432 or version != 2:
            print(f"FAIL: magic/version = {magic:#x}/{version}, want 0x616b3432/2 (llama2.c version 2, W8A8)")
            ok = False
        for k, v in EXPECTED.items():
            if got[k] != v:
                print(f"FAIL: {k} = {got[k]}, want {v}")
                ok = False

        # Layout check: in the interleaved layout, fp32 scales of the token
        # embedding come right after its int8 values. Scales are small positive numbers.
        off = 256 + 4 * (n_layers * dim * 2 + dim) + vocab_size * dim
        f.seek(off)
        scales = struct.unpack("8f", f.read(32))
        if not all(0.0 < s < 1.0 for s in scales):
            print("FAIL: data after the token embedding does not look like fp32 scales;"
                  " the file was probably written with the non-interleaved layout")
            ok = False

    size = os.path.getsize(path)
    if size != EXPECTED_SIZE:
        print(f"FAIL: file size = {size:,} bytes, want {EXPECTED_SIZE:,}")
        ok = False

    print(f"header: {got}, seq_len={seq_len}, shared_classifier={shared_classifier}")

    if "--sha256" in sys.argv:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        digest = h.hexdigest()
        same = digest == REFERENCE_SHA256
        print(f"sha256: {digest} ({'identical to' if same else 'differs from'} the paper checkpoint)")

    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
