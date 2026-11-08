"""v10: LightGBM training on the GPU when the runtime has one and it is faster - never required.

The LightGBM wheels on PyPI (4.x) carry the OpenCL GPU learner (`device_type='gpu'`) together with their own OpenCL
ICD loader, which looks for vendor drivers in /etc/OpenCL/vendors/*.icd (or the OCL_ICD_FILENAMES variable). Colab's
GPU runtimes ship the NVIDIA driver library (libnvidia-opencl.so.1) but no such entry, so `setup_opencl` writes it.
`choose` then times a short CPU fit and a short GPU fit of the same Task 2-like problem in a child process (an OpenCL
fault cannot take the notebook kernel down), checks that both reach the same fit quality, and takes the GPU only when
it is clearly faster. Every failure leaves training on the CPU. Only training moves: LightGBM predicts on the CPU.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

GPU_PARAMS = dict(device_type="gpu", gpu_use_dp=False)
LIB_DIRS = ("/usr/lib64-nvidia", "/usr/lib/x86_64-linux-gnu", "/usr/local/nvidia/lib64", "/usr/local/nvidia/lib",
            "/usr/lib64", "/usr/lib")
VENDORS = "/etc/OpenCL/vendors"


def nvidia_gpu() -> str | None:
    """'name, memory' of the first NVIDIA GPU as nvidia-smi reports it; None without one."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=60)
    except Exception:  # noqa: BLE001
        return None
    lines = [x.strip() for x in r.stdout.splitlines() if x.strip()]
    return lines[0] if r.returncode == 0 and lines else None


def _opencl_libs() -> list:
    libs = []
    for d in LIB_DIRS:
        libs += sorted(glob.glob(os.path.join(d, "libnvidia-opencl.so*")))
    if not libs:
        try:
            r = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=60)
            libs = [x.split("=>")[-1].strip() for x in r.stdout.splitlines() if "libnvidia-opencl.so" in x]
        except Exception:  # noqa: BLE001
            pass
    return libs


def setup_opencl() -> dict:
    """Register the NVIDIA OpenCL driver for the ICD loader inside the LightGBM wheel (idempotent)."""
    info = dict(libs=_opencl_libs())
    if not info["libs"]:
        info["note"] = "no libnvidia-opencl library found"
        return info
    lib = next((x for x in info["libs"] if x.endswith(".so.1")), info["libs"][0])
    have = sorted(glob.glob(os.path.join(VENDORS, "*.icd")))
    info["icd_before"] = have

    def _mentions_nvidia(f):
        try:
            return "nvidia" in open(f).read().lower()
        except OSError:
            return False

    if any(_mentions_nvidia(f) for f in have):
        return info
    try:
        os.makedirs(VENDORS, exist_ok=True)
        with open(os.path.join(VENDORS, "nvidia.icd"), "w") as f:
            f.write(lib + "\n")
        info["icd_written"] = os.path.join(VENDORS, "nvidia.icd")
    except OSError:   # no write access: the loader also reads this variable
        os.environ["OCL_ICD_FILENAMES"] = lib
        info["icd_env"] = lib
    return info


_BENCH = r'''
import json, sys, time
import numpy as np
import lightgbm as lgb
n, f, rounds, threads = (int(x) for x in sys.argv[1:5])
extra = json.loads(sys.argv[5])
rng = np.random.default_rng(0)
X = rng.standard_normal((n, f), dtype=np.float32)
z = 1.5 * X[:, 0] + X[:, 1] * X[:, 2] - np.abs(X[:, 3]) + 0.5 * np.sin(3 * X[:, 4]) + rng.standard_normal(n).astype(np.float32)
y = (z > 0.8).astype(np.float32)
X[rng.random((n, f)) < 0.05] = np.nan
p = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1, num_threads=threads, seed=0)
ne = min(n // 5, 200000)
out = {}
for dev in ("cpu", "gpu"):
    q = dict(p, **(extra if dev == "gpu" else {"device_type": "cpu"}))
    try:
        if dev == "gpu":   # the OpenCL kernels compile on first use: not timed
            lgb.train(q, lgb.Dataset(X[:20000], y[:20000], params=q), 2)
        ds = lgb.Dataset(X, y, params=q, free_raw_data=False)
        t0 = time.time()
        ds.construct()
        tc = time.time() - t0
        t0 = time.time()
        lgb.train(q, ds, 10)
        t10 = time.time() - t0
        t0 = time.time()
        m = lgb.train(q, ds, rounds)
        tr = time.time() - t0
        per = max((tr - t10) / max(rounds - 10, 1), 1e-6)   # the slope: start-up and data upload left out
        pr = np.clip(m.predict(X[:ne], num_threads=threads), 1e-7, 1 - 1e-7)
        ll = float(-np.mean(y[:ne] * np.log(pr) + (1 - y[:ne]) * np.log(1 - pr)))
        out[dev] = dict(seconds=round(tc + tr, 2), construct=round(tc, 2), fixed=round(max(t10 - 10 * per, 0.0), 3),
                        per_round=round(per, 5), logloss=round(ll, 5))
    except Exception as e:
        out[dev] = dict(error=type(e).__name__ + ": " + str(e)[:300])
print("BENCH_JSON " + json.dumps(out))
'''


def benchmark(n: int = 1_000_000, f: int = 100, rounds: int = 40, threads: int | None = None, extra: dict | None = None,
              timeout: int = 1500) -> dict:
    """CPU and GPU fit of the same problem in a child process: data binning (construct) and per-round training time
    separately, and the training log-loss."""
    threads = int(threads or os.cpu_count() or 2)
    extra = dict(GPU_PARAMS if extra is None else extra)
    with tempfile.TemporaryDirectory() as d:
        script = os.path.join(d, "lgb_bench.py")
        with open(script, "w") as fh:
            fh.write(_BENCH)
        try:
            r = subprocess.run([sys.executable, script, str(n), str(f), str(rounds), str(threads), json.dumps(extra)],
                               capture_output=True, text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return dict(error=f"benchmark did not finish in {timeout} s")
    for line in r.stdout.splitlines():
        if line.startswith("BENCH_JSON "):
            return json.loads(line[len("BENCH_JSON "):])
    return dict(error=f"benchmark process ended with code {r.returncode}: {(r.stderr or r.stdout)[-400:]}")


def speedup(b: dict, rounds: int = 500) -> float:
    """CPU time / GPU time of a fit with `rounds` boosting rounds (the binning is the same work on both)."""
    c, g = b["cpu"], b["gpu"]
    if "per_round" not in c or "per_round" not in g:   # an older stored benchmark: whole fits
        return c["seconds"] / max(g["seconds"], 1e-6)
    return ((c["construct"] + c.get("fixed", 0.0) + rounds * c["per_round"])
            / max(g["construct"] + g.get("fixed", 0.0) + rounds * g["per_round"], 1e-6))


def ok_bench(b: dict | None) -> bool:
    """A benchmark in which both fits ran (only such results are worth storing)."""
    return bool(b) and not b.get("error") and "seconds" in (b.get("cpu") or {}) and "seconds" in (b.get("gpu") or {})


def choose(mode: str = "auto", min_speedup: float = 1.25, bench: dict | None = None, **bench_kw) -> dict:
    """mode 'cpu': no GPU. 'gpu' or 'auto': set up OpenCL and benchmark; the GPU is used when it trains without error,
    reaches the CPU's fit quality and ('auto' only) is at least `min_speedup` times faster on a 500-round fit.
    `bench`: a stored benchmark result for this runtime (skips the timing). Returns {device, params (extra LightGBM
    training parameters, {} for the CPU), why, gpu, opencl, bench}."""
    out = dict(device="cpu", params={}, gpu=None, why="", opencl=None, bench=None)
    if mode == "cpu":
        out["why"] = "CPU requested"
        return out
    out["gpu"] = nvidia_gpu()
    if out["gpu"] is None:
        out["why"] = "no NVIDIA GPU in this runtime"
        return out
    out["opencl"] = setup_opencl()
    b = bench if bench else benchmark(extra=GPU_PARAMS, **bench_kw)
    out["bench"] = b
    cpu, gpu = (b or {}).get("cpu") or {}, (b or {}).get("gpu") or {}
    if b.get("error") or "seconds" not in gpu or "seconds" not in cpu:
        out["why"] = "GPU training did not work: " + str(b.get("error") or gpu.get("error") or cpu.get("error"))[:300]
        return out
    same = abs(gpu["logloss"] - cpu["logloss"]) <= 0.02 * cpu["logloss"] + 0.002
    speed = speedup(b)
    out["speedup"] = round(speed, 2)
    if not same:
        out["why"] = f"GPU fit quality differs from the CPU's (log-loss {gpu['logloss']} vs {cpu['logloss']})"
    elif mode == "auto" and speed < min_speedup:
        out["why"] = f"GPU only {speed:.2f}x as fast as the {os.cpu_count()} CPU cores: staying on the CPU"
    else:
        out.update(device="gpu", params=dict(GPU_PARAMS), why=f"GPU {speed:.2f}x as fast as the CPU, same fit quality")
    return out
