#!/usr/bin/env bash
# Auto-detects GPU (NVIDIA/AMD/Intel), CPU, or Apple Silicon
# and exports the appropriate vLLM backend/image

set -euo pipefail

echo "=== vLLM Hardware Detection ==="

BACKEND="cpu"
DOCKER_IMAGE="vllm/vllm-openai-cpu:v0.29.0"
DEVICE_FLAGS=""
EXTRA_ENV=""

# --- NVIDIA CUDA ---
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
  echo "[✓] NVIDIA GPU detected: $GPU_NAME (CUDA $CUDA_VERSION)"
  BACKEND="cuda"
  DOCKER_IMAGE="vllm/vllm-openai:v0.29.0"
  DEVICE_FLAGS="--gpus all"
  EXTRA_ENV="VLLM_WORKER_MULTIPROC_METHOD=spawn"

# --- AMD ROCm ---
elif command -v rocm-smi &>/dev/null && rocm-smi &>/dev/null 2>&1; then
  GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | head -1 || echo "AMD GPU")
  echo "[✓] AMD ROCm GPU detected: $GPU_NAME"
  BACKEND="rocm"
  DOCKER_IMAGE="vllm/vllm-openai-rocm:v0.29.0"
  DEVICE_FLAGS="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render"
  EXTRA_ENV="HSA_OVERRIDE_GFX_VERSION=11.0.0"

# --- Intel XPU ---
elif command -v xpu-smi &>/dev/null 2>&1 || ls /dev/dri/renderD* &>/dev/null 2>&1; then
  if lspci 2>/dev/null | grep -qi "intel.*graphics\|intel.*xe\|intel.*arc"; then
    echo "[✓] Intel XPU/Arc GPU detected"
    BACKEND="xpu"
    DOCKER_IMAGE="vllm/vllm-openai-xpu:v0.29.0"
    DEVICE_FLAGS="--device=/dev/dri"
    EXTRA_ENV="VLLM_USE_V1=1"
  fi
fi

# --- Apple Silicon (macOS) ---
if [[ "$BACKEND" == "cpu" ]] && [[ "$(uname)" == "Darwin" ]]; then
  ARCH=$(uname -m)
  if [[ "$ARCH" == "arm64" ]]; then
    echo "[✓] Apple Silicon (ARM64) detected — using CPU backend with Metal acceleration"
    BACKEND="apple"
    DOCKER_IMAGE=""  # Native install recommended on macOS
    EXTRA_ENV="VLLM_CPU_OMP_THREADS_BIND=all"
  fi
fi

# --- CPU fallback ---
if [[ "$BACKEND" == "cpu" ]]; then
  CPU_INFO=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown CPU")
  CORES=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo "4")
  echo "[!] No dedicated GPU found — using CPU backend"
  echo "    CPU: $CPU_INFO | Cores: $CORES"
  DOCKER_IMAGE="vllm/vllm-openai-cpu:v0.29.0"
  EXTRA_ENV="OMP_NUM_THREADS=${CORES} VLLM_CPU_OMP_THREADS_BIND=0-$((CORES-2))"
fi

echo ""
echo "--- Detected Configuration ---"
echo "  BACKEND      = $BACKEND"
echo "  DOCKER_IMAGE = $DOCKER_IMAGE"
echo "  DEVICE_FLAGS = $DEVICE_FLAGS"
echo ""

# Export for use by other scripts
export VLLM_BACKEND="$BACKEND"
export VLLM_DOCKER_IMAGE="$DOCKER_IMAGE"
export VLLM_DEVICE_FLAGS="$DEVICE_FLAGS"
export VLLM_EXTRA_ENV="$EXTRA_ENV"

# Write to .env file for docker-compose
cat > .env.detected << EOF
VLLM_BACKEND=${BACKEND}
VLLM_DOCKER_IMAGE=${DOCKER_IMAGE}
VLLM_DEVICE_FLAGS=${DEVICE_FLAGS}
VLLM_EXTRA_ENV=${EXTRA_ENV}
EOF

echo "[✓] Hardware detection complete. Written to .env.detected"
