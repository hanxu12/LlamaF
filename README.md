# LlamaF

An FPGA accelerator for Llama2-architecture LLM inference on embedded FPGAs.

This is the code for the paper **LlamaF: An Efficient Llama2 Architecture Accelerator on Embedded FPGAs**
(Han Xu, Yutong Li, Shihao Ji; IEEE WF-IoT 2024; [DOI](https://doi.org/10.1109/WF-IoT62078.2024.10811385) · [arXiv:2409.11424](https://arxiv.org/abs/2409.11424)).

## Design

- **Group-wise W8A8 quantization (GS = 256).** Weights are quantized after training, and activations
  at run time, to INT8 with one FP32 scale per group of 256 values. TinyLlama shrinks from 4.4 GB to 1.1 GB.
  RMSNorm weights stay in FP32.
- **Fully pipelined GQMV accelerator.** GQMV is group-wise quantized matrix-vector multiplication.
  The HLS design has three stages: pre-process, dot product (256-wide SIMD and an adder tree),
  and accumulate. The stages are connected by `hls::stream` and run under `#pragma HLS DATAFLOW`.
  There are two kernels:
  - kernel1 (`matmul1`): column size `dim` = 2048 (8 groups): W<sub>q</sub>+W<sub>k</sub>+W<sub>v</sub>,
    W<sub>o</sub>, W<sub>1</sub>+W<sub>3</sub>, and the classifier
  - kernel2 (`matmul2`): column size `hidden_dim` = 5632 (22 groups): W<sub>2</sub>
- **Asynchronous computation (task-level scheduling).** Layer weights are loaded into a 111.5 MB buffer
  one layer at a time. Layer 0 is loaded at program start. After that, the next weights are copied
  while the FPGA kernel runs, which hides the off-chip transfer time.
  Multi-head attention, RoPE, RMSNorm and SwiGLU run on the PS with OpenMP.

## Results (from the paper)

TinyLlama 1.1B on ZCU102. Greedy sampling on SQuAD questions, with EOS omitted.

| Method | GOPS | tok/s, step=64 | tok/s, step=128 | tok/s, step=256 | tok/s/W |
|---|---|---|---|---|---|
| ZCU102 PS (4× A53, OpenMP, same W8A8 model) | 0.201 | 0.0935 | 0.0933 | 0.0928 | 0.0480 |
| LlamaF, no scheduling | 4.696 | 0.936 | 0.915 | 0.853 | 0.207 |
| **LlamaF** | **4.696** (23.4×) | **1.478** (15.8×) | **1.424** (15.3×) | **1.328** (14.3×) | **0.291** (6.1×) |

- WikiText-2 perplexity: 7.05 (W32A32) → 7.09 (W8A8, GS=256), an increase of 0.57%.

## Repository layout

```
LlamaF/
├── host/
│   └── run.cpp                 # Host program (XRT): tokenizer, sampling, attention, RMSNorm, kernel scheduling
├── kernels/
│   ├── matmul1.cpp             # GQMV kernel1: column size dim = 2048 (8 groups)
│   └── matmul2.cpp             # GQMV kernel2: column size hidden_dim = 5632 (22 groups)
├── config/
│   ├── zcu102.cfg              # v++ configuration
│   └── xrt.ini                 # XRT runtime configuration
├── export/
│   ├── export.py               # Quantize a HuggingFace model to the W8A8 (GS = 256) checkpoint
│   ├── model.py                # Llama model definition used by export.py
│   └── requirements.txt
├── tokenizer/
│   └── tok_tl-chat.bin         # TinyLlama-Chat tokenizer (32003 tokens), llama2.c format
├── scripts/
│   ├── export_w8a8_gs256.sh    # Download TinyLlama and generate the W8A8 (GS = 256) checkpoint
│   ├── verify_bin.py           # Check a checkpoint's header, layout and size (no dependencies)
│   ├── build_hw.sh             # Build host app, kernels, xclbin and SD card image
│   └── run_hw.sh               # Run on the board
├── CITATION.cff
├── LICENSE
└── README.md
```

## Requirements

- Board: Xilinx ZCU102
- Tools: Vitis / Vitis HLS / Vivado 2022.2, aarch64-xilinx-linux-g++ 11.2.0,
  platform `xilinx_zcu102_base_202220_1`, ZynqMP common image 2022.2 (rootfs + kernel)
- Model export: Python 3 with the packages in `export/requirements.txt`

## 1. Generate the quantized model

```bash
pip install -r export/requirements.txt
scripts/export_w8a8_gs256.sh
# -> models/tinyllama-1.1b-chat-v0.2-w8a8-gs256.bin (1,117,524,416 bytes)
```

Model weights are not stored in this repository. The script downloads the original FP32 model from
HuggingFace, pinned to the revision used in the paper:

| | |
|---|---|
| Model | [`TinyLlama/TinyLlama-1.1B-Chat-v0.2`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.2) |
| Revision | [`c907b2fa`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.2/tree/c907b2fa3103c3646ca02870ee4b91ce4e260c31) (2023-09-17) |
| Size | 4.4 GB (FP32) → 1.1 GB after quantization |

The script downloads that model (only `model.safetensors`, the config and the tokenizer files),
quantizes it with group-wise W8A8, GS = 256 (`export.py --version 2`), and verifies the result.

To check that a file matches the checkpoint used in the paper byte for byte:

```bash
python3 scripts/verify_bin.py models/tinyllama-1.1b-chat-v0.2-w8a8-gs256.bin --sha256
```

Checkpoint format: llama2.c version 2 ("Q8_0", magic `ak42`). The fp32 RMSNorm weights come first, then
each quantized tensor as int8 values followed by its fp32 group scales. The classifier is not shared
with the token embedding.

## 2. Build

```bash
source <Vitis_2022.2>/settings64.sh
source <sysroot>/environment-setup-cortexa72-cortexa53-xilinx-linux   # sets CXX and SYSROOT
export ROOTFS=<path to xilinx-zynqmp-common-v2022.2>                 # has rootfs.ext4 and Image

scripts/build_hw.sh            # all steps (kernel synthesis takes several hours)
scripts/build_hw.sh host       # or rebuild only the host app after changing host/run.cpp
```

Output goes to `build/hw/`. The kernels run on the platform's default clock, **150 MHz**. This is the
clock of the xclbin used for the paper's measurements. The paper reports 205 MHz, which is the
Vitis HLS estimate of kernel1's maximum frequency, not the clock the kernels ran at.

Profiling and tracing are both off by default:

- **Hardware profiling** (`config/zcu102.cfg`): off. The xclbin used for the paper was built with
  profiling on (`debug=1` and `[profile] data=all:all:all`), which adds AXI and kernel monitors to the
  design. The monitors use extra LUTs and FFs and can make timing harder to meet, but they do not
  change the kernels.
- **Runtime tracing** (`config/xrt.ini`): off. The original configuration had `native_xrt_trace=true`, which logs
  every XRT API call on the host and adds a small run-time overhead.

To profile with `vitis_analyzer`, uncomment the profiling lines in `config/zcu102.cfg` and rebuild
the kernels, then enable tracing under `[Debug]` in `config/xrt.ini`, for example
`native_xrt_trace=true` and `device_trace=fine` (device tracing needs the profiling build).

## 3. Run on the ZCU102

1. Copy `build/hw/package/sd_card/*` to the SD card's boot partition.
2. Copy `models/tinyllama-1.1b-chat-v0.2-w8a8-gs256.bin` to the same partition.
3. Boot the board and run:

```sh
source /mnt/sd-mmcblk0p1/run_hw.sh "Once upon a time" 256
```

The program prints the generated text and `achieved tok/s` at the end.

`run_hw.sh` uses greedy decoding (`-t 0`), the setting used for the paper's measurements. To call the
program directly:

```sh
./app.exe tinyllama-1.1b-chat-v0.2-w8a8-gs256.bin -z tok_tl-chat.bin -t 0 -n 256 -i "Once upon a time"
```

Options: `-t` temperature (default 1.0; 0 = greedy), `-p` top-p (default 0.9), `-s` random seed,
`-n` number of steps, `-i` prompt, `-z` tokenizer path.

## Future improvements

These are not in the paper's design. The estimates are from a per-token time model
(CPU ≈ 237 ms, kernels ≈ 440 ms, weight copy ≈ 391 ms, with the copy overlapped with the kernels)
and have not been measured on the board.

| Improvement | What it changes |
|---|---|
| Higher kernel clock (200 / 300 MHz) | Map `x_stream16_arr` to BRAM with `#pragma HLS BIND_STORAGE` and cut the high-fanout nets around it. At 150 MHz the worst path is the write enable of this buffer, almost all routing delay. |
| Four AXI ports for the weights | Split each kernel's weight interface into four `m_axi` bundles and connect them to HP0–HP3 with `[connectivity] sp=...`. Today every kernel port shares HP0, so weights are limited to one 128-bit port (2.4 GB/s at 150 MHz) |
| Keep all weights in device buffers | Allocate buffers for all 22 layers (about 1 GB) and copy the weights once at start-up, instead of copying each layer every token. Needs a CMA region of at least about 1.1 GB on the board |

## Citation

If you use LlamaF in your research, please cite:

```bibtex
@inproceedings{xu2024llamaf,
  title     = {LlamaF: An Efficient Llama2 Architecture Accelerator on Embedded FPGAs},
  author    = {Xu, Han and Li, Yutong and Ji, Shihao},
  booktitle = {2024 IEEE 10th World Forum on Internet of Things (WF-IoT)},
  pages     = {1--7},
  year      = {2024},
  organization = {IEEE},
  doi       = {10.1109/WF-IoT62078.2024.10811385}
}
```

Preprint: [arXiv:2409.11424](https://arxiv.org/abs/2409.11424)

```bibtex
@article{xu2024llamaf_arxiv,
  title   = {LlamaF: An Efficient Llama2 Architecture Accelerator on Embedded FPGAs},
  author  = {Xu, Han and Li, Yutong and Ji, Shihao},
  journal = {arXiv preprint arXiv:2409.11424},
  year    = {2024}
}
```

## Acknowledgements

The host-side software of LlamaF is adapted from [llama2.c](https://github.com/karpathy/llama2.c)
by Andrej Karpathy (MIT License). The quantization scheme and the C inference approach come from that project:

| LlamaF file | Origin in llama2.c | What LlamaF changed |
|---|---|---|
| `host/run.cpp` | `runq.c`: int8 (Q8_0) inference in pure C. Group-wise symmetric INT8 quantization of weights and run-time quantization of activations, the checkpoint format and memory mapping, tokenizer, sampler, and the transformer forward pass | Offloads every quantized matrix-vector multiplication (GQMV) to the FPGA kernels through XRT. Concatenates W<sub>q</sub>/W<sub>k</sub>/W<sub>v</sub> and W<sub>1</sub>/W<sub>3</sub> into single kernel calls. Loads weights asynchronously while the kernel runs. Multi-head attention stays on the PS, parallelized with OpenMP as in `runq.c`. |
| `export/export.py` | `export.py`: `version2_export` (Q8_0 export) | GS = 256 by default. HuggingFace loading supports grouped-query attention (TinyLlama has 4 KV heads). |
| `export/model.py` | `model.py` | Unchanged |

The FPGA side (`kernels/matmul1.cpp`, `kernels/matmul2.cpp`: the pipelined GQMV accelerator) and
the ZCU102 build and run scripts are new in LlamaF.

TinyLlama is by Zhang et al. ([TinyLlama](https://github.com/jzhang38/TinyLlama), Apache 2.0).

## License

MIT. See [LICENSE](LICENSE), which also includes the MIT notice for the code adapted from llama2.c.
