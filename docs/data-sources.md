# Data Sources

Detailed breakdown of where each piece of data comes from and how it's collected.

## GPU Metrics

### pynvml (NVIDIA Management Library)

Initialized once at import time. Used for:

| Metric | API Call | Notes |
|--------|----------|-------|
| GPU name | `nvmlDeviceGetName()` | e.g., "NVIDIA GB10" |
| SM utilization | `nvmlDeviceGetUtilizationRates().gpu` | 0-100% |
| Memory bandwidth utilization | `nvmlDeviceGetUtilizationRates().memory` | 0-100% |
| VRAM used/total | `nvmlDeviceGetMemoryInfo()` | **Not supported on GB10** — falls back to process sum |
| Temperature | `nvmlDeviceGetTemperature()` | Celsius |
| Power draw | `nvmlDeviceGetPowerUsage()` | Milliwatts, converted to Watts |
| SM clock | `nvmlDeviceGetClockInfo(NVML_CLOCK_SM)` | MHz |
| Compute processes | `nvmlDeviceGetComputeRunningProcesses()` | PID + GPU memory per process |
| Graphics processes | `nvmlDeviceGetGraphicsRunningProcesses()` | PID + GPU memory per process |
| Process CPU% | `psutil.Process(pid).cpu_percent(interval=None)` | Non-blocking; returns 0.0 on first call per process |

### nvidia-smi dmon

Subprocess call: `nvidia-smi dmon -s u -c 1`

Provides a second opinion on SM% and memory bandwidth%, plus encoder/decoder utilization. The dmon value for SM% is preferred over pynvml's `gpu_util` when available.

### Process Name Resolution

For each GPU process PID, `psutil.Process(pid).cmdline()` is used to build a descriptive name:

- Worker processes with `::` in argv[0] (e.g., `VLLM::EngineCore`, `sglang::scheduler`) use that directly
- Launcher processes (e.g., `python3 -m sglang.launch_server --model Qwen/Qwen3-1.7B`) extract the `--model` flag value

## Host Metrics

### psutil

| Metric | API Call |
|--------|----------|
| CPU % | `psutil.cpu_percent(interval=None)` — non-blocking, uses delta from previous call |
| Core count | `psutil.cpu_count(logical=True)` |
| RAM used/total | `psutil.virtual_memory()` |
| TCP connections | `psutil.net_connections(kind="tcp")` — used for per-port connection counting |

## Container Metrics

### docker CLI

**Discovery:** `docker ps --format '{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Ports}}'`

Containers are filtered by matching image + name against `INFERENCE_PATTERNS`.

**Stats:** `docker stats --no-stream --format '{{.ID}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}\t{{.PIDs}}'`

Provides CPU%, memory, block I/O, and PID count. Network I/O from this source is cumulative and human-formatted, so it's not used for rate calculation.

### Docker Engine API (unix socket)

**Endpoint:** `GET /containers/{name}/stats?stream=false&one-shot=true`

**Connection:** Direct HTTP over `/var/run/docker.sock` using Python's `http.client` with a custom `AF_UNIX` socket.

Provides raw integer byte counters per network interface:
- `networks.eth0.rx_bytes` — total bytes received
- `networks.eth0.tx_bytes` — total bytes transmitted

These counters are fed into `RateTracker` to compute bytes/sec rates.

## Server Endpoint Probes

### Health Checks

| Server | Endpoint | Healthy Response |
|--------|----------|-----------------|
| vLLM | `GET /health` | 200 |
| SGLang | `GET /health` | 200 |
| NIM | `GET /v1/health/ready` | 200 |

### Model Information

| Server | Endpoint | Data Extracted |
|--------|----------|----------------|
| All | `GET /v1/models` | Model ID, `max_model_len` (vLLM/NIM) |
| SGLang | `GET /get_model_info` | `model_path`, `max_total_num_tokens` |

### Runtime Metrics

#### NIM / vLLM — Prometheus Format

| Server | Endpoint |
|--------|----------|
| NIM | `GET /v1/metrics` |
| vLLM | `GET /metrics` |

Scraped metrics:

| Prometheus Metric | Display Column | Type |
|-------------------|---------------|------|
| `vllm:num_requests_running` | RUN | Gauge |
| `vllm:num_requests_waiting` | PEND | Gauge |
| `vllm:kv_cache_usage_perc` | KV$ | Gauge (0-1 scaled to 0-100%) |
| `request_finish_total` | RPS | Counter → rate |
| `vllm:prompt_tokens_total` | IN/s | Counter → rate |
| `vllm:generation_tokens_total` | OUT/s | Counter → rate |

#### SGLang — Prometheus Format (requires `--enable-metrics`)

| Server | Endpoint |
|--------|----------|
| SGLang | `GET /metrics` |

Scraped metrics:

| Prometheus Metric | Display Column | Type |
|-------------------|---------------|------|
| `sglang:num_running_reqs` | RUN | Gauge |
| `sglang:num_queue_reqs` | PEND | Gauge |
| `sglang:token_usage` | KV$ | Gauge (0-1 scaled to 0-100%) |
| `sglang:prompt_tokens_total` | IN/s | Counter → rate |
| `sglang:generation_tokens_total` | OUT/s | Counter → rate |
| `sglang:gen_throughput` | OUT/s (fallback) | Gauge (live tok/s, used when counter rate unavailable) |
| `sglang:cache_hit_rate` | (collected) | Gauge |

#### SGLang — /get_load (fallback)

When `--enable-metrics` is not set, llm-top falls back to `/get_load`:

| Field | Display Column |
|-------|---------------|
| `num_reqs` | RUN |
| `num_waiting_reqs` | PEND |

In fallback mode, KV$, RPS, IN/s, and OUT/s are unavailable.

## Rate Computation

All cumulative counters (Prometheus counters and Docker byte counters) are passed through `RateTracker.rate()`:

```
rate = (current_value - previous_value) / (current_time - previous_time)
```

- First refresh: returns `None` (displayed as `-`)
- Subsequent refreshes: returns the per-second rate
- Minimum elapsed time between samples: 0.5s (prevents division by near-zero)
- Negative deltas (counter resets): clamped to 0.0
