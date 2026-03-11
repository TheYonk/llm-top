# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`llm-top` is a single-file Python 3 live terminal dashboard for monitoring LLM inference servers on NVIDIA DGX Spark (GB10 / Blackwell). It combines GPU, host, container, and inference server metrics into a `top`-like display using `rich`.

## Running

```bash
pip install -r requirements.txt
python3 llm-top.py               # 2s refresh (default)
python3 llm-top.py --refresh 5   # custom refresh interval
```

Press `q` to quit. There is no build step, test suite, or linter configured.

## Architecture

The entire tool is `llm-top.py` (~1050 lines), organized as layered data collectors feeding into a single `build_dashboard()` orchestrator:

1. **GPU metrics** — `get_gpu_summary()`, `get_gpu_processes()`, `get_nvidia_smi_dmon()` via pynvml + `nvidia-smi dmon` subprocess
2. **Host metrics** — `get_host_summary()` via psutil
3. **Container discovery/metrics** — `discover_containers()`, `get_container_stats()`, `get_container_net_counters()` via Docker CLI + Docker Engine API (unix socket)
4. **Server probes** — `identify_server_type()`, `probe_model_info()`, `scrape_server_metrics()` via HTTP to vLLM/SGLang/NIM endpoints
5. **RateTracker** — generic `(key, counter_name) → delta/dt` calculator for any cumulative counter
6. **Display helpers** — color functions (`_color_pct`, `_color_temp`, `_color_health`) and formatters (`_fmt_rate`, `_fmt_tok_rate`, etc.)
7. **Main loop** — argparse, `rich.live.Live()` display, `RawTerminal` for non-blocking keyboard input

Data flow: `main()` → `build_dashboard()` → collectors (pynvml, psutil, docker, HTTP) → `RateTracker` → rich Table grid.

## Key Design Constraints

- **Graceful degradation**: Every pynvml call has its own try/except. On GB10, `nvmlDeviceGetMemoryInfo` is unsupported but other metrics work. Each field falls back to `None` / `-` independently.
- **Zero config**: Containers auto-discovered by regex matching image/name against `INFERENCE_PATTERNS`. Server types disambiguated via container hints + endpoint fingerprinting (SGLang has `/get_model_info`, vLLM doesn't).
- **Network rates from Docker API**: Raw byte counters fetched via unix socket (`/var/run/docker.sock`) instead of parsing `docker stats` human-formatted output.
- **Single file**: All code lives in `llm-top.py`. No packages, modules, or build system.

## System Dependencies

- NVIDIA GPU with driver and `nvidia-smi`
- Docker CLI and socket at `/var/run/docker.sock`
- Python packages: `rich`, `psutil`, `requests`, `nvidia-ml-py` (pynvml)

## Server Requirements

- **SGLang**: Launch with `--enable-metrics` to expose Prometheus metrics at `/metrics`. Without it, only basic request counts (RUN/PEND) are available via `/get_load` fallback.

## Documentation

- `docs/architecture.md` — module layout and data flow diagram
- `docs/data-sources.md` — where each metric comes from
- `docs/metrics-reference.md` — every column/field explained with color thresholds
