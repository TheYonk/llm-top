#!/usr/bin/env python3
"""llm-top: Live terminal dashboard for LLM inference servers on DGX Spark.

Monitors GPU utilization, container resources, model health, and active
connections for vLLM, SGLang, and NIM inference servers.

Usage:
    python3 llm-top.py               # default 2s refresh
    python3 llm-top.py --refresh 5   # 5s refresh
"""

import argparse
import json
import re
import select
import shutil
import subprocess
import sys
import termios
import time
import tty
from datetime import datetime

import psutil
import requests
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

# ── pynvml setup ──────────────────────────────────────────────────────────────

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False

# ── Known inference server patterns ───────────────────────────────────────────

# Image substrings / container name patterns used to identify inference containers
INFERENCE_PATTERNS = [
    "vllm",
    "sglang",
    "nim",
    "triton",
    "trt-llm",
    "lmdeploy",
]

# Known server types and their health/model endpoints
SERVER_PROBES = {
    "vLLM": {
        "health": "/health",
        "models": "/v1/models",
    },
    "SGLang": {
        "health": "/health",
        "models": "/v1/models",
        "extra": "/get_model_info",
    },
    "NIM": {
        "health": "/v1/health/ready",
        "models": "/v1/models",
    },
}


# ── GPU metrics ───────────────────────────────────────────────────────────────


def get_gpu_summary() -> dict:
    """Return per-GPU summary from pynvml with graceful per-field fallbacks.

    Some pynvml calls (e.g. nvmlDeviceGetMemoryInfo) are not supported on
    all platforms (GB10/Blackwell).  Each metric is fetched independently so
    a single unsupported call doesn't blank the whole GPU row.
    """
    if not _NVML_OK:
        return {}
    try:
        count = pynvml.nvmlDeviceGetCount()
    except Exception:
        return {}

    gpus = {}
    for i in range(count):
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
        except Exception:
            continue

        name = "GPU"
        try:
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
        except Exception:
            pass

        gpu_util, mem_util = None, None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            gpu_util = util.gpu
            mem_util = util.memory
        except Exception:
            pass

        mem_used, mem_total = None, None
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            mem_used = mem.used
            mem_total = mem.total
        except Exception:
            pass

        temp = None
        try:
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            pass

        power = None
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000  # mW -> W
        except Exception:
            pass

        clock = None
        try:
            clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        except Exception:
            pass

        gpus[i] = {
            "name": name,
            "gpu_util": gpu_util,
            "mem_util": mem_util,
            "mem_used": mem_used,
            "mem_total": mem_total,
            "temp": temp,
            "power": power,
            "clock": clock,
        }
    return gpus


def get_gpu_processes() -> list[dict]:
    """Return per-process GPU memory and type via pynvml."""
    procs = []
    if not _NVML_OK:
        return procs
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                infos = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            except Exception:
                infos = []
            for p in infos:
                name = _proc_name(p.pid)
                procs.append(
                    {
                        "pid": p.pid,
                        "gpu": i,
                        "mem": p.usedGpuMemory,
                        "name": name,
                        "type": "Compute",
                        "cpu": _proc_cpu(p.pid),
                    }
                )
            try:
                gfx = pynvml.nvmlDeviceGetGraphicsRunningProcesses(h)
            except Exception:
                gfx = []
            seen = {p.pid for p in infos}
            for p in gfx:
                if p.pid not in seen:
                    procs.append(
                        {
                            "pid": p.pid,
                            "gpu": i,
                            "mem": p.usedGpuMemory,
                            "name": _proc_name(p.pid),
                            "type": "Graphics",
                            "cpu": _proc_cpu(p.pid),
                        }
                    )
    except Exception:
        pass
    # Prune cache entries for PIDs no longer in the GPU process list.
    live_pids = {p["pid"] for p in procs}
    for dead in set(_proc_cache) - live_pids:
        _proc_cache.pop(dead, None)
    return procs


def _proc_name(pid: int) -> str:
    """Best-effort descriptive name for a GPU process."""
    try:
        p = psutil.Process(pid)
        cmdline = p.cmdline()
        if not cmdline:
            return p.name()

        cmd0 = cmdline[0]

        # Worker processes already have descriptive names like "VLLM::EngineCore"
        if "::" in cmd0:
            return cmd0

        # For launcher commands (python3 -m sglang.launch_server --model X),
        # extract the module + model name
        full = " ".join(cmdline).lower()
        model = _extract_flag(cmdline, "--model") or _extract_flag(cmdline, "--model-path")

        if "sglang" in full:
            tag = f"sglang ({model})" if model else "sglang"
            return tag
        if "vllm" in full:
            tag = f"vLLM ({model})" if model else "vLLM"
            return tag
        if "triton" in full:
            return "triton"

        return cmd0.split("/")[-1]
    except Exception:
        return f"<pid {pid}>"


_proc_cache: dict[int, psutil.Process] = {}


def _proc_cpu(pid: int) -> float | None:
    """Return CPU% for a process, or None on failure.

    psutil.Process.cpu_percent(interval=None) returns 0 on the first call for
    a given Process object — it needs two samples to compute a delta.  We cache
    Process objects by PID so the measurement accumulates across refresh cycles.
    """
    try:
        proc = _proc_cache.get(pid)
        if proc is None:
            proc = psutil.Process(pid)
            _proc_cache[pid] = proc
        return proc.cpu_percent(interval=None)
    except Exception:
        _proc_cache.pop(pid, None)
        return None


def _extract_flag(cmdline: list[str], flag: str) -> str | None:
    """Extract the value following a CLI flag like --model."""
    try:
        idx = cmdline.index(flag)
        if idx + 1 < len(cmdline):
            return cmdline[idx + 1].split("/")[-1]  # just the model name part
    except ValueError:
        pass
    return None


def get_nvidia_smi_dmon() -> dict:
    """Parse nvidia-smi dmon for SM%, mem BW%, encoder/decoder utilisation."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "dmon", "-s", "u", "-c", "1"],
            timeout=4,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        result = {}
        for line in out.strip().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 5:
                idx = int(parts[0])
                result[idx] = {
                    "sm": _safe_int(parts[1]),
                    "mem_bw": _safe_int(parts[2]),
                    "enc": _safe_int(parts[3]),
                    "dec": _safe_int(parts[4]),
                }
        return result
    except Exception:
        return {}


def _safe_int(s: str) -> int | None:
    try:
        v = int(s)
        return v if v >= 0 else None
    except ValueError:
        return None


# ── Host metrics ──────────────────────────────────────────────────────────────


def get_host_summary() -> dict:
    cpu_pct = psutil.cpu_percent(interval=None)
    cores = psutil.cpu_count(logical=True)
    mem = psutil.virtual_memory()
    return {
        "cpu_pct": cpu_pct,
        "cores": cores,
        "mem_used_gib": mem.used / (1024**3),
        "mem_total_gib": mem.total / (1024**3),
    }


# ── Container discovery & metrics ────────────────────────────────────────────


def discover_containers() -> list[dict]:
    """Return list of running containers that look like inference servers."""
    try:
        out = subprocess.check_output(
            [
                "docker",
                "ps",
                "--format",
                "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Ports}}",
            ],
            timeout=5,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    containers = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        cid, name, image = parts[0], parts[1], parts[2]
        ports_str = parts[3] if len(parts) > 3 else ""
        tag = image.lower() + " " + name.lower()
        if any(pat in tag for pat in INFERENCE_PATTERNS):
            containers.append(
                {
                    "id": cid,
                    "name": name,
                    "image": image,
                    "ports": _parse_host_ports(ports_str),
                }
            )
    return containers


def _parse_host_ports(ports_str: str) -> list[int]:
    """Extract host-side port numbers from docker ps Ports column."""
    ports = set()
    for m in re.finditer(r"0\.0\.0\.0:(\d+)->", ports_str):
        ports.add(int(m.group(1)))
    for m in re.finditer(r":::(\d+)->", ports_str):
        ports.add(int(m.group(1)))
    return sorted(ports)


def get_container_stats(container_ids: list[str]) -> dict[str, dict]:
    """Run docker stats --no-stream for the given container IDs."""
    if not container_ids:
        return {}
    try:
        out = subprocess.check_output(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.ID}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}\t{{.PIDs}}",
            ]
            + container_ids,
            timeout=10,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}

    stats = {}
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        cid = parts[0]
        stats[cid] = {
            "cpu": parts[1],
            "mem": parts[2],
            "net": parts[3],
            "block": parts[4],
            "pids": parts[5],
        }
    return stats


def get_container_logs(container_name: str, lines: int = 50) -> list[str]:
    """Fetch the last N log lines from a container via docker logs."""
    try:
        out = subprocess.check_output(
            ["docker", "logs", "--tail", str(lines), container_name],
            timeout=5,
            text=True,
            stderr=subprocess.STDOUT,
        )
        return out.splitlines()
    except Exception:
        return []


def get_container_net_counters(container_names: list[str]) -> dict[str, dict]:
    """Fetch raw network byte counters per container via Docker API.

    Returns {container_name: {"rx_bytes": int, "tx_bytes": int}}.
    """
    result = {}
    import http.client
    import socket

    class DockerSocket(http.client.HTTPConnection):
        def __init__(self):
            super().__init__("localhost")

        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect("/var/run/docker.sock")
            self.sock.settimeout(3)

    try:
        conn = DockerSocket()
    except Exception:
        return result

    for name in container_names:
        try:
            conn.request("GET", f"/containers/{name}/stats?stream=false&one-shot=true")
            resp = conn.getresponse()
            if resp.status != 200:
                resp.read()
                continue
            data = json.loads(resp.read())
            nets = data.get("networks", {})
            rx = sum(iface.get("rx_bytes", 0) for iface in nets.values())
            tx = sum(iface.get("tx_bytes", 0) for iface in nets.values())
            result[name] = {"rx_bytes": rx, "tx_bytes": tx}
        except Exception:
            pass

    try:
        conn.close()
    except Exception:
        pass
    return result


# ── Server endpoint probes ────────────────────────────────────────────────────


def identify_server_type(port: int, hint: str = "") -> tuple[str, dict] | None:
    """Identify which server type is running on a port.

    Args:
        port: The port to probe.
        hint: Container name/image string to help disambiguate (lowercase).
    """
    base = f"http://localhost:{port}"

    # If we have a container hint, try the matching server type first
    hint = hint.lower()
    order = list(SERVER_PROBES.keys())
    if "sglang" in hint:
        order = ["SGLang"] + [k for k in order if k != "SGLang"]
    elif "vllm" in hint:
        order = ["vLLM"] + [k for k in order if k != "vLLM"]
    elif "nim" in hint:
        order = ["NIM"] + [k for k in order if k != "NIM"]

    for stype in order:
        endpoints = SERVER_PROBES[stype]
        try:
            r = requests.get(base + endpoints["health"], timeout=1.5)
            if r.status_code == 200:
                # Extra check: SGLang has /get_model_info, vLLM doesn't
                if stype == "vLLM" and "sglang" not in hint:
                    try:
                        r2 = requests.get(base + "/get_model_info", timeout=1.0)
                        if r2.status_code == 200:
                            return "SGLang", SERVER_PROBES["SGLang"]
                    except Exception:
                        pass
                return stype, endpoints
        except Exception:
            continue

    # Fallback: try /v1/models which all OpenAI-compat servers have
    try:
        r = requests.get(base + "/v1/models", timeout=1.5)
        if r.status_code == 200:
            return "OpenAI-compat", {"models": "/v1/models"}
    except Exception:
        pass
    return None


def probe_model_info(port: int, server_type: str, endpoints: dict) -> dict:
    """Fetch model name, max context, health status from a server."""
    base = f"http://localhost:{port}"
    info: dict = {
        "server": server_type,
        "port": port,
        "health": "unknown",
        "model": "-",
        "max_ctx": None,
    }

    # Health
    health_ep = endpoints.get("health")
    if health_ep:
        try:
            r = requests.get(base + health_ep, timeout=1.5)
            info["health"] = "ok" if r.status_code == 200 else f"{r.status_code}"
        except Exception:
            info["health"] = "down"

    # Models list
    models_ep = endpoints.get("models")
    if models_ep:
        try:
            r = requests.get(base + models_ep, timeout=1.5)
            if r.status_code == 200:
                data = r.json()
                models = data.get("data", [])
                if models:
                    info["model"] = models[0].get("id", "-")
                    # vLLM includes max_model_len
                    info["max_ctx"] = models[0].get("max_model_len")
        except Exception:
            pass

    # SGLang extra info
    if server_type == "SGLang" and "extra" in endpoints:
        try:
            r = requests.get(base + endpoints["extra"], timeout=1.5)
            if r.status_code == 200:
                d = r.json()
                if "max_total_num_tokens" in d:
                    info["max_ctx"] = d["max_total_num_tokens"]
                if "model_path" in d and info["model"] == "-":
                    info["model"] = d["model_path"]
        except Exception:
            pass

    # NIM health label
    if server_type == "NIM" and info["health"] == "ok":
        info["health"] = "ready"

    return info


def count_connections(port: int) -> int:
    """Count ESTABLISHED TCP connections to a local port."""
    count = 0
    try:
        for conn in psutil.net_connections(kind="tcp"):
            if conn.status == "ESTABLISHED" and conn.laddr and conn.laddr.port == port:
                count += 1
    except (psutil.AccessDenied, PermissionError):
        pass
    return count


# ── Server runtime metrics (Prometheus / SGLang) ─────────────────────────────

# Prometheus metric names we care about
_PROM_GAUGES = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:token_usage",
    "sglang:gen_throughput",
    "sglang:cache_hit_rate",
]
_PROM_COUNTERS = [
    "request_finish_total",
    "sglang:num_requests_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
]

# Metrics endpoints per server type
_METRICS_ENDPOINTS = {
    "NIM": "/v1/metrics",
    "vLLM": "/metrics",
    "SGLang": "/metrics",
}


def _parse_prom_line(line: str) -> tuple[str, float] | None:
    """Parse a single Prometheus exposition line into (metric_name, value)."""
    if line.startswith("#") or not line.strip():
        return None
    # Strip label portion: name{labels} value  or  name value
    m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?[^}]*\}?\s+(-?[\d.eE+\-]+)", line)
    if m:
        try:
            return m.group(1), float(m.group(2))
        except ValueError:
            pass
    return None


def scrape_server_metrics(port: int, server_type: str) -> dict:
    """Scrape runtime metrics from an inference server.

    Returns a dict with keys:
        running, pending, kv_cache_pct,
        total_reqs, prompt_tokens, gen_tokens,
        gen_throughput, cache_hit_rate
    All values may be None if unavailable.
    """
    result = {
        "running": None,
        "pending": None,
        "kv_cache_pct": None,
        "total_reqs": None,
        "prompt_tokens": None,
        "gen_tokens": None,
        "gen_throughput": None,
        "cache_hit_rate": None,
    }
    base = f"http://localhost:{port}"

    # ── Try Prometheus /metrics for all server types ──
    metrics_ep = _METRICS_ENDPOINTS.get(server_type)
    if metrics_ep:
        try:
            r = requests.get(base + metrics_ep, timeout=2.0)
            if r.status_code == 200:
                wanted = set(_PROM_GAUGES + _PROM_COUNTERS)
                for line in r.text.splitlines():
                    parsed = _parse_prom_line(line)
                    if not parsed:
                        continue
                    name, val = parsed
                    if name not in wanted:
                        continue
                    # vLLM / NIM metrics
                    if name == "vllm:num_requests_running":
                        result["running"] = val
                    elif name == "vllm:num_requests_waiting":
                        result["pending"] = val
                    elif name == "vllm:kv_cache_usage_perc":
                        result["kv_cache_pct"] = val * 100  # 0-1 -> 0-100
                    elif name == "vllm:prompt_tokens_total":
                        result["prompt_tokens"] = val
                    elif name == "vllm:generation_tokens_total":
                        result["gen_tokens"] = val
                    # SGLang metrics
                    elif name == "sglang:num_running_reqs":
                        result["running"] = val
                    elif name == "sglang:num_queue_reqs":
                        result["pending"] = val
                    elif name == "sglang:token_usage":
                        result["kv_cache_pct"] = val * 100  # 0-1 -> 0-100
                    elif name == "sglang:prompt_tokens_total":
                        result["prompt_tokens"] = val
                    elif name == "sglang:generation_tokens_total":
                        result["gen_tokens"] = val
                    elif name == "sglang:gen_throughput":
                        result["gen_throughput"] = val
                    elif name == "sglang:cache_hit_rate":
                        result["cache_hit_rate"] = val
                    # Shared counters
                    elif name == "request_finish_total":
                        result["total_reqs"] = val
                    elif name == "sglang:num_requests_total":
                        result["total_reqs"] = val
                # If Prometheus yielded data, we're done
                if any(v is not None for v in result.values()):
                    return result
        except Exception:
            pass

    # ── SGLang fallback: /get_load + /get_server_info (when --enable-metrics is off) ──
    if server_type == "SGLang":
        try:
            r = requests.get(base + "/get_load", timeout=1.5)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    d = data[0]
                else:
                    d = data
                result["running"] = d.get("num_reqs")
                result["pending"] = d.get("num_waiting_reqs")
        except Exception:
            pass
        try:
            r = requests.get(base + "/get_server_info", timeout=1.5)
            if r.status_code == 200:
                d = r.json()
                # Live stats are nested inside internal_states[0]
                states = d.get("internal_states", [])
                inner = states[0] if states else d
                if "last_gen_throughput" in inner:
                    result["gen_throughput"] = inner["last_gen_throughput"]
                mem = inner.get("memory_usage", {})
                if mem and "token_capacity" in mem:
                    # Store token capacity for display (not a % yet, but available)
                    result["_token_capacity"] = mem["token_capacity"]
        except Exception:
            pass

    return result


class RateTracker:
    """Generic rate calculator for cumulative counters.

    Stores the previous (timestamp, value) for any (key, counter_name) pair
    and computes delta/elapsed_seconds on the next call.
    """

    def __init__(self):
        # key -> {counter_name: (timestamp, value)}
        self._prev: dict[str, dict[str, tuple[float, float]]] = {}

    def rate(self, key: str, counter_name: str, current_value: float | None) -> float | None:
        """Record a cumulative counter value and return its per-second rate.

        Returns None on the first call (no previous sample) or if current_value
        is None.
        """
        if current_value is None:
            return None
        now = time.time()
        prev_counters = self._prev.setdefault(key, {})
        result = None
        if counter_name in prev_counters:
            prev_t, prev_v = prev_counters[counter_name]
            dt = now - prev_t
            if dt > 0.5:
                result = max(0.0, (current_value - prev_v) / dt)
        prev_counters[counter_name] = (now, current_value)
        return result


# ── Display helpers ───────────────────────────────────────────────────────────


def _color_pct(value: int | float | None, low: int = 50, high: int = 85) -> Text:
    """Color a percentage value green/yellow/red."""
    if value is None:
        return Text("-", style="dim")
    s = f"{value:.0f}%" if isinstance(value, float) else f"{value}%"
    if value >= high:
        return Text(s, style="bold red")
    if value >= low:
        return Text(s, style="yellow")
    return Text(s, style="green")


def _color_temp(temp: int | None) -> Text:
    if temp is None:
        return Text("-", style="dim")
    s = f"{temp}C"
    if temp >= 85:
        return Text(s, style="bold red")
    if temp >= 70:
        return Text(s, style="yellow")
    return Text(s, style="green")


def _color_health(h: str) -> Text:
    h_lower = h.lower()
    if h_lower in ("ok", "ready"):
        return Text(h, style="bold green")
    if h_lower == "unknown":
        return Text(h, style="yellow")
    return Text(h, style="bold red")


def _fmt_rate(bps: float | None) -> str:
    """Format a bytes-per-second rate into a human-readable string."""
    if bps is None:
        return "-"
    if bps >= 1024**3:
        return f"{bps / 1024**3:.1f} GB/s"
    if bps >= 1024**2:
        return f"{bps / 1024**2:.1f} MB/s"
    if bps >= 1024:
        return f"{bps / 1024:.1f} KB/s"
    return f"{bps:.0f} B/s"


def _fmt_tok_rate(tps: float | None) -> str:
    """Format a tokens-per-second rate."""
    if tps is None:
        return "-"
    if tps >= 1000:
        return f"{tps / 1000:.1f}k"
    return f"{tps:.1f}"


def _fmt_count(v: float | None) -> str:
    """Format an integer counter, or '-' if None."""
    if v is None:
        return "-"
    return str(int(v))


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "-"
    if n >= 1024**3:
        return f"{n / 1024**3:,.1f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:,.0f} MiB"
    return f"{n / 1024:,.0f} KiB"


# ── Build the log view ───────────────────────────────────────────────────────


def build_log_view(
    container_name: str, refresh_sec: float, index: int = 0, total: int = 1
) -> Table:
    """Build a full-screen log view for a container."""
    now = datetime.now().strftime("%H:%M:%S")

    # Use terminal height to decide how many log lines to show
    term_height = shutil.get_terminal_size((80, 40)).lines
    # Reserve lines for header (2) + footer (2)
    log_lines_count = max(10, term_height - 4)

    logs = get_container_logs(container_name, lines=log_lines_count)

    outer = Table.grid(padding=(0, 0))
    outer.add_column(ratio=1)

    # Header
    header = Text()
    header.append("llm-top", style="bold cyan")
    header.append(" — LOGS: ", style="dim")
    header.append(container_name, style="bold white")
    if total > 1:
        header.append(f"  ({index + 1}/{total})", style="bold yellow")
    header.append(f"    Refresh: {refresh_sec:.0f}s    ", style="dim")
    header.append(now, style="bold white")
    outer.add_row(header)
    outer.add_row(Text(""))

    # Log lines
    for line in logs:
        outer.add_row(Text(line, style="white", overflow="ellipsis", no_wrap=True))

    # Footer
    outer.add_row(Text(""))
    tab_hint = "Tab next | " if total > 1 else ""
    outer.add_row(Text(f"{tab_hint}D dashboard | q quit", style="dim italic"))

    return outer


# ── Build the dashboard ──────────────────────────────────────────────────────


def build_dashboard(refresh_sec: float, tracker: RateTracker | None = None) -> Table:
    """Collect all metrics and assemble a rich Table layout."""
    now = datetime.now().strftime("%H:%M:%S")

    # ── Gather data ──
    gpus = get_gpu_summary()
    dmon = get_nvidia_smi_dmon()
    host = get_host_summary()
    gpu_procs = get_gpu_processes()
    containers = discover_containers()
    cstats = get_container_stats([c["id"] for c in containers])
    net_counters = get_container_net_counters([c["name"] for c in containers])

    # Compute network rates per container
    container_net_rates: dict[str, dict] = {}
    if tracker:
        for c in containers:
            nc = net_counters.get(c["name"])
            if nc:
                key = f"ctr:{c['name']}"
                container_net_rates[c["name"]] = {
                    "rx_s": tracker.rate(key, "rx_bytes", nc["rx_bytes"]),
                    "tx_s": tracker.rate(key, "tx_bytes", nc["tx_bytes"]),
                }

    # Probe all exposed ports from inference containers
    probed_ports: set[int] = set()
    model_infos: list[dict] = []
    for c in containers:
        for port in c["ports"]:
            if port in probed_ports:
                continue
            probed_ports.add(port)
            hint = c["name"] + " " + c["image"]
            result = identify_server_type(port, hint=hint)
            if result:
                stype, endpoints = result
                info = probe_model_info(port, stype, endpoints)
                info["connections"] = count_connections(port)
                info["container"] = c["name"]
                # Scrape runtime metrics (queue depth, KV cache, token counts)
                metrics = scrape_server_metrics(port, stype)
                info["metrics"] = metrics
                # Compute per-second rates from cumulative counters
                rates: dict[str, float | None] = {}
                if tracker:
                    key = f"srv:{port}"
                    rates["rps"] = tracker.rate(key, "total_reqs", metrics.get("total_reqs"))
                    rates["prompt_tok_s"] = tracker.rate(key, "prompt_tokens", metrics.get("prompt_tokens"))
                    rates["gen_tok_s"] = tracker.rate(key, "gen_tokens", metrics.get("gen_tokens"))
                info["rates"] = rates
                model_infos.append(info)

    # ── Outer layout table (no borders, full width) ──
    outer = Table.grid(padding=(0, 0))
    outer.add_column(ratio=1)

    # ── Header ──
    gpu_name = next(iter(gpus.values()), {}).get("name", "GPU")
    header = Text()
    header.append("llm-top", style="bold cyan")
    header.append(f" — {gpu_name}    ", style="dim")
    header.append(f"Refresh: {refresh_sec:.0f}s    ", style="dim")
    header.append(now, style="bold white")
    outer.add_row(header)
    outer.add_row(Text(""))

    # ── GPU summary line(s) ──
    # If pynvml mem is unavailable, estimate from process list
    gpu_mem_used_fallback = sum(p.get("mem") or 0 for p in gpu_procs)

    for idx, g in gpus.items():
        d = dmon.get(idx, {})
        sm = d.get("sm") if d.get("sm") is not None else g["gpu_util"]
        mem_bw = d.get("mem_bw")
        line = Text()
        line.append("GPU ", style="bold")
        line.append(" | ", style="dim")
        line.append_text(_color_pct(sm, low=40, high=80))
        line.append(" SM", style="dim")
        line.append(" | ", style="dim")
        line.append_text(_color_pct(mem_bw, low=40, high=80) if mem_bw is not None else Text("-", style="dim"))
        line.append(" Mem BW", style="dim")
        line.append(" | ", style="dim")
        line.append_text(_color_temp(g["temp"]))
        if g["power"] is not None:
            line.append(" | ", style="dim")
            line.append(f"{g['power']:.0f}W", style="white")
        if g["clock"]:
            line.append(" | ", style="dim")
            line.append(f"{g['clock']} MHz", style="white")
        # Memory: use pynvml if available, else sum from processes
        mem_used = g["mem_used"] if g["mem_used"] is not None else gpu_mem_used_fallback
        mem_total = g["mem_total"]
        if mem_used or mem_total:
            line.append(" | ", style="dim")
            used_str = _fmt_bytes(mem_used) if mem_used else "~" + _fmt_bytes(gpu_mem_used_fallback)
            total_str = _fmt_bytes(mem_total) if mem_total else "?"
            line.append(f"{used_str} / {total_str}", style="white")
        outer.add_row(line)

    # ── Host summary ──
    host_line = Text()
    host_line.append("HOST", style="bold")
    host_line.append(" | ", style="dim")
    host_line.append_text(_color_pct(host["cpu_pct"], low=50, high=85))
    host_line.append(f" CPU ({host['cores']} cores)", style="dim")
    host_line.append(" | ", style="dim")
    host_line.append(
        f"{host['mem_used_gib']:.1f} / {host['mem_total_gib']:.1f} GiB RAM",
        style="white",
    )
    outer.add_row(host_line)
    outer.add_row(Text(""))

    # ── GPU Processes ──
    if gpu_procs:
        pt = Table(title="PROCESSES (GPU)", title_style="bold", expand=True, padding=(0, 1))
        pt.add_column("PID", style="cyan", width=10)
        pt.add_column("NAME", min_width=30)
        pt.add_column("CPU%", justify="right", width=8)
        pt.add_column("GPU MEM", justify="right", width=12)
        pt.add_column("TYPE", width=10)
        # sort by GPU mem descending
        gpu_procs.sort(key=lambda p: p.get("mem") or 0, reverse=True)
        for p in gpu_procs:
            cpu = p.get("cpu")
            cpu_str = _color_pct(cpu, low=50, high=80) if cpu is not None else Text("-", style="dim")
            pt.add_row(
                str(p["pid"]),
                Text(p["name"], style="white"),
                cpu_str,
                _fmt_bytes(p["mem"]),
                p["type"],
            )
        outer.add_row(pt)
        outer.add_row(Text(""))

    # ── Containers ──
    if containers:
        ct = Table(title="CONTAINERS", title_style="bold", expand=True, padding=(0, 1))
        ct.add_column("NAME", min_width=22)
        ct.add_column("CPU%", justify="right", width=10)
        ct.add_column("MEM", justify="right", width=14)
        ct.add_column("RX/s", justify="right", width=10)
        ct.add_column("TX/s", justify="right", width=10)
        ct.add_column("BLOCK I/O", justify="right", width=18)
        ct.add_column("PIDS", justify="right", width=6)
        for c in containers:
            s = cstats.get(c["id"], {})
            nr = container_net_rates.get(c["name"], {})
            ct.add_row(
                Text(c["name"], style="white"),
                s.get("cpu", "-"),
                s.get("mem", "-"),
                _fmt_rate(nr.get("rx_s")),
                _fmt_rate(nr.get("tx_s")),
                s.get("block", "-"),
                s.get("pids", "-"),
            )
        outer.add_row(ct)
        outer.add_row(Text(""))

    # ── Models ──
    if model_infos:
        mt = Table(title="MODELS", title_style="bold", expand=True, padding=(0, 1))
        mt.add_column("SERVER", width=10)
        mt.add_column("MODEL", min_width=18)
        mt.add_column("PORT", justify="right", width=6)
        mt.add_column("HEALTH", width=7)
        mt.add_column("RUN", justify="right", width=5)
        mt.add_column("PEND", justify="right", width=5)
        mt.add_column("KV$", justify="right", width=6)
        mt.add_column("CACHE", justify="right", width=6)
        mt.add_column("RPS", justify="right", width=7)
        mt.add_column("IN/s", justify="right", width=9)
        mt.add_column("OUT/s", justify="right", width=9)
        mt.add_column("CONNS", justify="right", width=6)
        for m in model_infos:
            mx = m.get("metrics", {})
            rates = m.get("rates", {})

            # Running / pending requests
            run = _fmt_count(mx.get("running"))
            pend = _fmt_count(mx.get("pending"))

            # KV cache
            kv = mx.get("kv_cache_pct")
            kv_text = _color_pct(kv, low=50, high=85) if kv is not None else Text("-", style="dim")

            # Prefix cache hit rate
            chr = mx.get("cache_hit_rate")
            if chr is not None:
                cache_text = _color_pct(chr * 100, low=30, high=70) if chr <= 1.0 else _color_pct(chr, low=30, high=70)
            else:
                cache_text = Text("-", style="dim")

            # Per-second rates (counter-based, or SGLang gauge fallback)
            rps = rates.get("rps")
            rps_text = f"{rps:.1f}" if rps is not None else "-"
            in_tok_s = rates.get("prompt_tok_s")
            in_text = _fmt_tok_rate(in_tok_s)
            out_tok_s = rates.get("gen_tok_s")
            # SGLang provides gen_throughput as a live gauge (tok/s)
            if out_tok_s is None and mx.get("gen_throughput") is not None:
                out_tok_s = mx["gen_throughput"]
            out_text = _fmt_tok_rate(out_tok_s)

            mt.add_row(
                Text(m["server"], style="bold"),
                m["model"],
                str(m["port"]),
                _color_health(m["health"]),
                run,
                pend,
                kv_text,
                cache_text,
                rps_text,
                in_text,
                out_text,
                str(m["connections"]),
            )
        outer.add_row(mt)

    # ── Footer ──
    outer.add_row(Text(""))
    outer.add_row(Text("L logs | q quit", style="dim italic"))

    return outer


# ── Keyboard input (non-blocking) ────────────────────────────────────────────


class RawTerminal:
    """Context manager to set terminal to raw/cbreak mode for key detection."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = None

    def __enter__(self):
        try:
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        except Exception:
            self.old_settings = None
        return self

    def __exit__(self, *args):
        if self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def key_pressed(self) -> str | None:
        """Return a character if one is available, else None."""
        try:
            if select.select([self.fd], [], [], 0)[0]:
                return sys.stdin.read(1)
        except Exception:
            pass
        return None


# ── Main loop ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="llm-top: live terminal dashboard for LLM inference servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        help="Refresh interval in seconds (default: 2)",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Dump a single snapshot to stdout as plain text and exit",
    )
    args = parser.parse_args()

    console = Console()
    tracker = RateTracker()

    # Kick psutil CPU measurement (first call returns 0)
    psutil.cpu_percent(interval=None)

    if args.text:
        text_console = Console(force_terminal=False, no_color=True)
        text_console.print(build_dashboard(args.refresh, tracker))
        return

    view_mode = "dashboard"  # "dashboard" or "logs"
    log_containers = []  # list of container names for log view
    log_index = 0  # current index into log_containers

    def _handle_key(k, view_mode, log_containers, log_index):
        """Handle a key press, return updated (view_mode, log_containers, log_index, quit, changed)."""
        if k == "q":
            return view_mode, log_containers, log_index, True, False
        if k == "l" and view_mode == "dashboard":
            containers = discover_containers()
            if containers:
                log_containers = [c["name"] for c in containers]
                log_index = 0
                view_mode = "logs"
                return view_mode, log_containers, log_index, False, True
        elif k == "d" and view_mode == "logs":
            view_mode = "dashboard"
            log_containers = []
            log_index = 0
            return view_mode, log_containers, log_index, False, True
        elif k == "\t" and view_mode == "logs" and len(log_containers) > 1:
            log_index = (log_index + 1) % len(log_containers)
            return view_mode, log_containers, log_index, False, True
        return view_mode, log_containers, log_index, False, False

    with RawTerminal() as term:
        with Live(
            build_dashboard(args.refresh, tracker),
            console=console,
            refresh_per_second=1,
            screen=True,
        ) as live:
            try:
                while True:
                    # Check for keys
                    key = term.key_pressed()
                    if key:
                        view_mode, log_containers, log_index, quit_, _ = _handle_key(
                            key.lower() if key != "\t" else key,
                            view_mode, log_containers, log_index,
                        )
                        if quit_:
                            break

                    if view_mode == "logs" and log_containers:
                        live.update(build_log_view(
                            log_containers[log_index], args.refresh,
                            index=log_index, total=len(log_containers),
                        ))
                    else:
                        live.update(build_dashboard(args.refresh, tracker))

                    # Sleep in small increments so we can catch key presses
                    slept = 0.0
                    while slept < args.refresh:
                        time.sleep(0.1)
                        slept += 0.1
                        key = term.key_pressed()
                        if key:
                            view_mode, log_containers, log_index, quit_, changed = _handle_key(
                                key.lower() if key != "\t" else key,
                                view_mode, log_containers, log_index,
                            )
                            if quit_:
                                return
                            if changed:
                                break  # re-render immediately
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
