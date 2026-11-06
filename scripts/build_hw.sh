#!/usr/bin/env bash
# Build LlamaF for the ZCU102 (hardware target): host app, HLS kernels, xclbin, SD card image.
#
# Requires a Vitis 2022.2 environment:
#   source <Vitis>/settings64.sh
#   source <PetaLinux sysroot>/environment-setup-cortexa72-cortexa53-xilinx-linux   # sets CXX, SYSROOT
#   export ROOTFS=<xilinx-zynqmp-common-v2022.2>   # dir with rootfs.ext4 and Image
#
#   Usage: scripts/build_hw.sh [host|kernels|package|all]   (default: all)
#
# Synthesis and implementation of the kernels take several hours.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLATFORM="${PLATFORM:-xilinx_zcu102_base_202220_1}"
BUILD="$ROOT/build/hw"
CFG="$ROOT/config/zcu102.cfg"
STEP="${1:-all}"

: "${CXX:?set CXX (source the PetaLinux SDK environment-setup script)}"
: "${SYSROOT:?set SYSROOT to the aarch64 sysroot}"

mkdir -p "$BUILD"
cd "$BUILD"

build_host() {
    echo "== host: app.exe"
    $CXX -g -std=c++17 -Wall -O3 -fmessage-length=0 "$ROOT/host/run.cpp" -o ./app.exe \
        -I"$SYSROOT/usr/include/xrt" -L"$SYSROOT/usr/lib" -lxrt_coreutil -pthread \
        --sysroot="$SYSROOT" -fopenmp
}

build_kernels() {
    echo "== kernels: matmul1.xo, matmul2.xo -> vadd.xclbin"
    v++ -c -t hw --platform "$PLATFORM" --config "$CFG" -k matmul1 -I"$ROOT/kernels" \
        "$ROOT/kernels/matmul1.cpp" -o ./matmul1.xo
    v++ -c -t hw --platform "$PLATFORM" --config "$CFG" -k matmul2 -I"$ROOT/kernels" \
        "$ROOT/kernels/matmul2.cpp" -o ./matmul2.xo
    # host/run.cpp loads "./vadd.xclbin", so keep that name.
    # No clock option: the kernels run on the platform's default clock (id 0, 150 MHz), as in the paper.
    # The platform's fixed clocks are id 0: 150, 1: 300, 2: 75, 3: 100, 4: 200, 5: 400, 6: 600 MHz.
    # To try another one, add e.g. --clock.id=4:matmul1_1,matmul2_1 (200 MHz) here, and compile the
    # kernels for that clock with --hls.clock=200000000:matmul1 (and matmul2) in the v++ -c steps.
    # Timing closure above 150 MHz has not been verified on xczu9eg.
    v++ -l -t hw --platform "$PLATFORM" --config "$CFG" ./matmul1.xo ./matmul2.xo -o ./vadd.xclbin
}

build_package() {
    : "${ROOTFS:?set ROOTFS to the dir with rootfs.ext4 and Image}"
    echo "== package: sd_card"
    cp "$ROOT/config/xrt.ini" "$ROOT/scripts/run_hw.sh" .
    v++ -p -t hw --platform "$PLATFORM" --config "$CFG" ./vadd.xclbin \
        --package.out_dir ./package --package.rootfs "$ROOTFS/rootfs.ext4" \
        --package.sd_file "$ROOTFS/Image" --package.sd_file ./xrt.ini \
        --package.sd_file ./app.exe --package.sd_file ./vadd.xclbin \
        --package.sd_file ./run_hw.sh --package.sd_file "$ROOT/tokenizer/tok_tl-chat.bin"
    echo "Copy build/hw/package/sd_card/* to the SD card's boot partition,"
    echo "then copy the model (models/tinyllama-1.1b-chat-v0.2-w8a8-gs256.bin, 1.1 GB) next to it."
}

case "$STEP" in
    host)    build_host ;;
    kernels) build_kernels ;;
    package) build_package ;;
    all)     build_host; build_kernels; build_package ;;
    *)       echo "unknown step: $STEP (host|kernels|package|all)"; exit 1 ;;
esac
