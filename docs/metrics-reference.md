# Metrics Reference

Every column and value displayed by `llm-top`, what it means, and how it's color-coded.

## Header Line

```
llm-top — NVIDIA GB10    Refresh: 2s    05:53:37
```

- **GPU name** — from pynvml
- **Refresh** — configured interval
- **Time** — current wall clock

## GPU Row

```
GPU  | 96% SM | 0% Mem BW | 70C | 37W | 2450 MHz | 84.5 GiB / ?
```

| Field | Description | Color Thresholds |
|-------|-------------|-----------------|
| **SM** | Streaming multiprocessor utilization | green <40%, yellow 40-80%, red >80% |
| **Mem BW** | Memory bandwidth utilization | green <40%, yellow 40-80%, red >80% |
| **Temperature** | GPU die temperature (Celsius) | green <70C, yellow 70-85C, red >85C |
| **Power** | Current power draw (Watts) | White (no threshold) |
| **Clock** | Current SM clock frequency (MHz) | White (no threshold) |
| **VRAM** | Used / Total GPU memory | White. Shows `~X / ?` when `nvmlDeviceGetMemoryInfo` is unsupported (GB10) — the `~` prefix indicates the value is estimated from process memory sum |

## HOST Row

```
HOST | 11% CPU (20 cores) | 97.3 / 119.7 GiB RAM
```

| Field | Description | Color Thresholds |
|-------|-------------|-----------------|
| **CPU** | System-wide CPU utilization | green <50%, yellow 50-85%, red >85% |
| **cores** | Logical CPU count | — |
| **RAM** | Used / Total system memory (GiB) | White (no threshold) |

## PROCESSES (GPU)

| Column | Description |
|--------|-------------|
| **PID** | Linux process ID |
| **NAME** | Descriptive process name. Worker processes show their argv[0] (e.g., `VLLM::EngineCore`). Launcher processes show `server (model)` format (e.g., `sglang (Qwen3-1.7B)`) |
| **GPU MEM** | GPU memory allocated to this process (GiB/MiB) |
| **TYPE** | `Compute` or `Graphics` |

Sorted by GPU memory descending.

## CONTAINERS

| Column | Description | Source |
|--------|-------------|--------|
| **NAME** | Docker container name | `docker ps` |
| **CPU%** | Container CPU utilization (can exceed 100% on multi-core) | `docker stats` |
| **MEM** | Container memory usage / host limit | `docker stats` |
| **RX/s** | Network receive rate (bytes/sec) | Docker API raw counters, delta/dt |
| **TX/s** | Network transmit rate (bytes/sec) | Docker API raw counters, delta/dt |
| **BLOCK I/O** | Cumulative block device read / write | `docker stats` |
| **PIDS** | Number of processes in the container | `docker stats` |

Network rates are formatted with adaptive units: `B/s`, `KB/s`, `MB/s`, `GB/s`.

## MODELS

| Column | Description | Source | Availability |
|--------|-------------|--------|--------------|
| **SERVER** | Detected server type | Endpoint probing | All |
| **MODEL** | Model ID or path | `/v1/models` | All |
| **PORT** | Host port the server listens on | `docker ps` port mapping | All |
| **HEALTH** | Server health status | Health endpoint probe | All |
| **RUN** | Currently running (in-flight) requests | Prometheus gauge / SGLang `/get_load` fallback | All |
| **PEND** | Queued/waiting requests | Prometheus gauge / SGLang `/get_load` fallback | All |
| **KV$** | KV cache utilization percentage | `vllm:kv_cache_usage_perc` / `sglang:token_usage` | All (SGLang requires `--enable-metrics`) |
| **RPS** | Completed requests per second | `request_finish_total` delta/dt | NIM, vLLM |
| **IN/s** | Prompt tokens processed per second | `vllm:prompt_tokens_total` / `sglang:prompt_tokens_total` delta/dt | All (SGLang requires `--enable-metrics`) |
| **OUT/s** | Generation tokens produced per second | `vllm:generation_tokens_total` delta/dt, or `sglang:gen_throughput` gauge | All (SGLang requires `--enable-metrics`) |
| **CONNS** | Active TCP connections to the server port | `psutil.net_connections()` | All |

### Health Status Colors

| Value | Color | Meaning |
|-------|-------|---------|
| `ok` | Green | Server healthy |
| `ready` | Green | NIM-specific healthy status |
| `unknown` | Yellow | Could not determine |
| Anything else | Red | Unhealthy or HTTP error code |

### KV Cache Colors

| Range | Color |
|-------|-------|
| 0-50% | Green |
| 50-85% | Yellow |
| >85% | Red |

### Token Rate Formatting

| Range | Format | Example |
|-------|--------|---------|
| < 1000 tok/s | `X.Y` | `42.5` |
| >= 1000 tok/s | `X.Yk` | `1.2k` |

### Rate Column Behavior

All rate columns (`RX/s`, `TX/s`, `RPS`, `IN/s`, `OUT/s`) show `-` on the first refresh cycle. They need two consecutive samples to compute `delta / elapsed_seconds`. After the first interval, rates update live on every refresh.

When there is no active inference traffic, rate columns show `0.0`.

### SGLang Metrics

SGLang exposes Prometheus metrics at `/metrics` when launched with `--enable-metrics`. The following SGLang-specific metrics are scraped:

| Prometheus Metric | Maps To | Type |
|-------------------|---------|------|
| `sglang:num_running_reqs` | RUN | Gauge |
| `sglang:num_queue_reqs` | PEND | Gauge |
| `sglang:token_usage` | KV$ | Gauge (0-1 scaled to %) |
| `sglang:prompt_tokens_total` | IN/s | Counter (delta/dt) |
| `sglang:generation_tokens_total` | OUT/s | Counter (delta/dt) |
| `sglang:gen_throughput` | OUT/s (fallback) | Gauge (live tok/s) |
| `sglang:cache_hit_rate` | (collected) | Gauge |

If `--enable-metrics` is not set, llm-top falls back to SGLang's `/get_load` endpoint which only provides RUN and PEND.
