"""
Python-based hardware detector for vLLM launcher.
Detects NVIDIA CUDA, AMD ROCm, Intel XPU, or CPU backend.
"""

import subprocess
import platform
import os
import sys
from dataclasses import dataclass
from enum import Enum


class Backend(str, Enum):
    CUDA   = "cuda"
    ROCM   = "rocm"
    XPU    = "xpu"
    CPU    = "cpu"
    APPLE  = "apple"


@dataclass
class HardwareConfig:
    backend: Backend
    device_name: str
    dtype: str
    extra_args: list[str]
    docker_image: str
    env_vars: dict[str, str]


def run(cmd: str) -> tuple[int, str]:
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return result.returncode, result.stdout.strip()
    except Exception:
        return 1, ""


def detect_hardware() -> HardwareConfig:
    """Auto-detect the best available backend."""

    # --- NVIDIA CUDA ---
    code, out = run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if code == 0 and out:
        gpu_name = out.split(",")[0].strip()
        vram_mb  = out.split(",")[1].strip() if "," in out else "unknown"
        print(f"[✓] NVIDIA GPU: {gpu_name} | VRAM: {vram_mb}")
        # Choose dtype based on VRAM
        vram_val = int(vram_mb.split()[0]) if vram_mb[0].isdigit() else 8000
        dtype = "bfloat16" if vram_val >= 8000 else "float16"
        return HardwareConfig(
            backend=Backend.CUDA,
            device_name=gpu_name,
            dtype=dtype,
            extra_args=[
                "--enable-prefix-caching",
                "--enable-chunked-prefill",
                "--gpu-memory-utilization", "0.90",
                "--max-num-batched-tokens", "16384",
                "--speculative-model", "auto",  # auto spec decode if supported
            ],
            docker_image="vllm/vllm-openai:v0.29.0",
            env_vars={
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            },
        )

    # --- AMD ROCm ---
    code, out = run("rocm-smi --showproductname")
    if code == 0 and out:
        print(f"[✓] AMD ROCm GPU: {out[:60]}")
        return HardwareConfig(
            backend=Backend.ROCM,
            device_name=out[:60],
            dtype="bfloat16",
            extra_args=[
                "--enable-prefix-caching",
                "--enable-chunked-prefill",
                "--gpu-memory-utilization", "0.90",
            ],
            docker_image="vllm/vllm-openai-rocm:v0.29.0",
            env_vars={"HIP_VISIBLE_DEVICES": "0"},
        )

    # --- Intel Arc/XPU ---
    code, out = run("lspci")
    if code == 0 and any(k in out.lower() for k in ["intel arc", "intel xe", "intel graphics"]):
        print("[✓] Intel XPU/Arc GPU detected")
        return HardwareConfig(
            backend=Backend.XPU,
            device_name="Intel XPU",
            dtype="bfloat16",
            extra_args=[
                "--enable-prefix-caching",
                "--gpu-memory-utilization", "0.85",
            ],
            docker_image="vllm/vllm-openai-xpu:v0.29.0",
            env_vars={"VLLM_USE_V1": "1"},
        )

    # --- Apple Silicon ---
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("[✓] Apple Silicon (M-series) detected — CPU/Metal backend")
        import multiprocessing
        cores = multiprocessing.cpu_count()
        return HardwareConfig(
            backend=Backend.APPLE,
            device_name="Apple Silicon",
            dtype="bfloat16",
            extra_args=["--device", "cpu", "--dtype", "bfloat16"],
            docker_image="",  # native install preferred
            env_vars={
                "VLLM_CPU_OMP_THREADS_BIND": f"0-{cores - 2}",
                "OMP_NUM_THREADS": str(cores - 1),
            },
        )

    # --- CPU fallback ---
    import multiprocessing
    cores = multiprocessing.cpu_count()
    # Reserve 2 cores for the HTTP server
    inference_cores = max(1, cores - 2)
    print(f"[!] No GPU found — CPU backend ({cores} cores, {inference_cores} for inference)")
    return HardwareConfig(
        backend=Backend.CPU,
        device_name=f"CPU ({cores} cores)",
        dtype="bfloat16",
        extra_args=[
            "--device", "cpu",
            "--dtype", "bfloat16",
        ],
        docker_image="vllm/vllm-openai-cpu:v0.29.0",
        env_vars={
            "OMP_NUM_THREADS": str(inference_cores),
            "VLLM_CPU_OMP_THREADS_BIND": f"0-{inference_cores - 1}",
            "VLLM_CPU_KVCACHE_SPACE": "8",  # GB of RAM for KV cache
        },
    )


if __name__ == "__main__":
    hw = detect_hardware()
    print(f"\nBackend  : {hw.backend}")
    print(f"Device   : {hw.device_name}")
    print(f"dtype    : {hw.dtype}")
    print(f"Image    : {hw.docker_image}")
    print(f"Env vars : {hw.env_vars}")
