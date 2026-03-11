# llm-top

A `top`-like live terminal dashboard for monitoring LLM inference servers running on NVIDIA DGX Spark.

```
llm-top — NVIDIA GB10    Refresh: 2s    05:53:37

GPU  | 96% SM | 0% Mem BW | 70C | 37W | 2450 MHz | 84.5 GiB / ?
HOST | 11% CPU (20 cores) | 97.3 / 119.7 GiB RAM

PROCESSES (GPU)
PID         NAME                        GPU MEM   TYPE
2501586     VLLM::EngineCore           76.0 GiB   Compute
2503338     sglang::scheduler           8.4 GiB   Compute
2503061     sglang (Qwen3-1.7B)         170 MiB   Compute

CONTAINERS
NAME                    CPU%       MEM         RX/s        TX/s       BLOCK I/O         PIDS
sglang-qwen3-1.7b    101.46%   4.379GiB    536 B/s     450 B/s    1.48GB / 338MB      228
nim-nemotron-3-nano     2.56%   3.876GiB    481 B/s    9.1 KB/s   1.04GB / 404MB      127

MODELS
SERVER     MODEL                    PORT  HEALTH  RUN  PEND  KV$    RPS    IN/s    OUT/s  CONNS
SGLang     Qwen/Qwen3-1.7B        30000  ok        1     0    -      -       -       -      0
NIM        nvidia/nemotron-3-nano   8006  ready     0     0   0%    0.0     0.0     0.0      0
```

## Quick Start

```bash
python3 llm-top.py               # 2s refresh (default)
python3 llm-top.py --refresh 5   # 5s refresh
```

Press `q` to quit.

## What It Monitors

| Section | Data | Source |
|---------|------|--------|
| **GPU** | SM utilization, memory bandwidth, temperature, power, clock, VRAM | `pynvml`, `nvidia-smi dmon` |
| **HOST** | CPU %, core count, RAM used/total | `psutil` |
| **PROCESSES** | Per-process GPU memory and type for all GPU compute/graphics processes | `pynvml` |
| **CONTAINERS** | CPU %, memory, network RX/TX rates, block I/O, PID count | `docker stats`, Docker API |
| **MODELS** | Server type, model name, health, running/pending requests, KV cache %, RPS, token throughput (in/out), active connections | Server HTTP endpoints, Prometheus metrics |

## Supported Inference Servers

| Server | Health | Models | Metrics | Queue Depth |
|--------|--------|--------|---------|-------------|
| **vLLM** | `/health` | `/v1/models` | `/metrics` (Prometheus) | running/waiting requests, KV cache %, RPS, tok/s |
| **SGLang** | `/health` | `/v1/models`, `/get_model_info` | `/get_load` | running/waiting requests |
| **NIM** | `/v1/health/ready` | `/v1/models` | `/v1/metrics` (Prometheus) | running/waiting requests, KV cache %, RPS, tok/s |

Containers are auto-discovered by matching image/name against known patterns: `vllm`, `sglang`, `nim`, `triton`, `trt-llm`, `lmdeploy`.

## Rate Metrics

All cumulative counters are displayed as per-second rates computed from the delta between consecutive refreshes:

- **RX/s, TX/s** — container network throughput (bytes/sec from Docker API raw counters)
- **RPS** — completed requests per second
- **IN/s** — prompt tokens processed per second
- **OUT/s** — generation tokens produced per second

The first refresh shows `-` for rates (needs two samples to compute a delta).

## Prerequisites

### Python Packages

```bash
pip install -r requirements.txt
```

| Package | PyPI Name | Min Version | Purpose |
|---------|-----------|-------------|---------|
| `rich` | `rich` | 13.0 | Terminal UI (Live display, Tables, colored Text) |
| `psutil` | `psutil` | 5.9 | Host CPU/RAM, TCP connection counting |
| `requests` | `requests` | 2.28 | HTTP probes to inference server endpoints |
| `pynvml` | `nvidia-ml-py` | 12.0 | GPU metrics (utilization, temp, power, per-process VRAM) |

All four are pre-installed on DGX Spark.

### System Requirements

| Requirement | Why | Check |
|-------------|-----|-------|
| `nvidia-smi` | SM/memory bandwidth via `dmon` | `which nvidia-smi` |
| `docker` CLI | Container discovery and stats | `which docker` |
| Docker socket (`/var/run/docker.sock`) | Raw network byte counters for rate computation | `ls -la /var/run/docker.sock` |
| NVIDIA GPU with driver | pynvml initialization | `nvidia-smi` |

The Docker socket must be readable by the user running `llm-top`. On most setups this means being in the `docker` group or running as root.

## Platform Notes

- Tested on DGX Spark (NVIDIA GB10 / Blackwell SM 12.x)
- `nvmlDeviceGetMemoryInfo` is not supported on GB10 — VRAM usage is estimated from the sum of per-process GPU memory
- Each pynvml metric is fetched independently with fallbacks so unsupported calls don't break the display

## Docs

- [Architecture](docs/architecture.md) — how the tool is structured
- [Data Sources](docs/data-sources.md) — detailed breakdown of where each metric comes from
- [Metrics Reference](docs/metrics-reference.md) — every column explained
