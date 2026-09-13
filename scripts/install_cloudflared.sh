#!/usr/bin/env bash
# Installs cloudflared CLI on Linux (x86_64 / arm64)
set -euo pipefail

ARCH=$(uname -m)
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

echo "[→] Installing cloudflared for $OS/$ARCH..."

if [[ "$OS" == "linux" ]]; then
  # Add Cloudflare GPG key and apt repo
  curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
    | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null

  echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] \
    https://pkg.cloudflare.com/cloudflared $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/cloudflared.list

  sudo apt-get update && sudo apt-get install -y cloudflared
  echo "[✓] cloudflared installed: $(cloudflared --version)"

elif [[ "$OS" == "darwin" ]]; then
  brew install cloudflare/cloudflare/cloudflared
  echo "[✓] cloudflared installed: $(cloudflared --version)"

else
  echo "[✗] Unsupported OS: $OS"
  exit 1
fi
