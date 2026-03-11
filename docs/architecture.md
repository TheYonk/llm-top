# Architecture

`llm-top` is a single-file Python script organized into layered data collectors, a rate tracker, display helpers, and a main refresh loop.

## Module Layout

```
llm-top.py
├── pynvml setup           # init once at import time
├── Constants               # INFERENCE_PATTERNS, SERVER_PROBES
├── GPU metrics             # get_gpu_summary(), get_gpu_processes(), get_nvidia_smi_dmon()
├── Host metrics            # get_host_summary()
├── Container discovery     # discover_containers(), get_container_stats(), get_container_net_counters()
├── Server probes           # identify_server_type(), probe_model_info(), scrape_server_metrics()
├── RateTracker             # generic delta/dt calculator for cumulative counters
├── Display helpers         # _color_pct(), _fmt_rate(), _fmt_tok_rate(), etc.
├── build_dashboard()       # orchestrates all collectors, assembles rich Tables
├── RawTerminal             # non-blocking keyboard input via termios
└── main()                  # argparse, Live display loop, quit detection
```

## Data Flow

```
                 ┌─────────────┐
                 │  main loop  │
                 │  (2s tick)  │
                 └──────┬──────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  build_dashboard()  │
              └──────┬──────────────┘
                     │
        ┌────────────┼────────────┬───────────────┐
        ▼            ▼            ▼               ▼
   ┌─────────┐ ┌──────────┐ ┌──────────┐  ┌──────────────┐
   │  pynvml │ │  psutil  │ │  docker  │  │  HTTP probes │
   │  + dmon │ │          │ │  CLI+API │  │  per server  │
   └────┬────┘ └────┬─────┘ └────┬─────┘  └──────┬───────┘
        │           │            │                │
        ▼           ▼            ▼                ▼
   GPU summary  Host summary  Container      Model info
   GPU procs    CPU / RAM     stats + net     + metrics
                              counters        + queue depth
                                    │                │
                                    ▼                ▼
                              ┌──────────────────────────┐
                              │      RateTracker         │
                              │  delta / elapsed → /s    │
                              └──────────┬───────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │   rich Table layout   │
                              │   GPU | HOST          │
                              │   PROCESSES           │
                              │   CONTAINERS          │
                              │   MODELS              │
                              └──────────────────────┘
```

## Key Design Decisions

### Graceful Degradation per Metric

Each pynvml call is wrapped in its own try/except. On the DGX Spark GB10, `nvmlDeviceGetMemoryInfo` is not supported but utilization, temperature, power, and clock all work fine. Rather than losing the entire GPU row, each field falls back to `None` and the display shows `-` or estimates from process data.

### Container Discovery by Pattern Matching

Instead of requiring configuration, containers are discovered by matching their Docker image name and container name against a list of known inference server patterns (`vllm`, `sglang`, `nim`, `triton`, `trt-llm`, `lmdeploy`). This is zero-config for common setups.

### Server Type Disambiguation

vLLM and SGLang both respond to `/health`, so a naive probe order would misidentify SGLang as vLLM. The tool uses two strategies:
1. **Container hint** — if the container name/image contains `sglang`, try SGLang endpoints first
2. **Endpoint fingerprinting** — if a port responds to `/health` but also has `/get_model_info`, it's SGLang (vLLM doesn't expose that endpoint)

### Generic RateTracker

Rather than hardcoding rate computation per metric, `RateTracker` is a generic `(key, counter_name) → rate` calculator. Any cumulative counter from any source (Prometheus, Docker API, etc.) can be tracked by calling `tracker.rate(key, name, value)`. This makes it trivial to add new rate-based metrics.

### Network Rates via Docker API

`docker stats` provides human-formatted cumulative totals (e.g., "3.34MB / 791kB") which are difficult to parse back into numbers for delta computation. Instead, raw byte counters are fetched directly from the Docker Engine API via the unix socket (`/var/run/docker.sock`), giving exact integer values suitable for rate calculation.

### Non-blocking Keyboard Input

The `RawTerminal` context manager sets the terminal to cbreak mode. The sleep loop between refreshes checks for keypresses every 100ms, allowing responsive `q` to quit without blocking the display.
