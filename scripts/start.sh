#!/usr/bin/env bash
# Main entrypoint — detects hardware and launches vLLM + cloudflared
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Load user config
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

MODEL_ID="${MODEL_ID:-microsoft/Phi-3.5-mini-instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
CF_TUNNEL_TOKEN="${CF_TUNNEL_TOKEN:-}"

echo "============================================"
echo "  vLLM Server — Auto Hardware Detection"
echo "============================================"
echo "  Model    : $MODEL_ID"
echo "  Port     : $VLLM_PORT"
echo ""

# Detect hardware and export vars
source "$SCRIPT_DIR/detect_hardware.sh"

# Start Cloudflare tunnel in background (if token set)
if [[ -n "$CF_TUNNEL_TOKEN" ]]; then
  echo "[→] Starting Cloudflare Tunnel..."
  cloudflared tunnel --no-autoupdate run \
    --token "$CF_TUNNEL_TOKEN" &
  CF_PID=$!
  echo "[✓] Cloudflare Tunnel PID: $CF_PID"
else
  echo "[!] CF_TUNNEL_TOKEN not set — skipping Cloudflare Tunnel"
  echo "    Set it in .env to expose your endpoint publicly"
fi

# Launch vLLM
echo "[→] Launching vLLM server..."
python "$ROOT_DIR/src/launcher.py" --model "$MODEL_ID"
