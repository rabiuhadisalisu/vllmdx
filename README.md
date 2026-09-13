# vLLM Server — Auto Hardware Detection + Cloudflare Tunnel

A self-hosted LLM server powered by **vLLM v0.29.0** with:
- ✅ Auto hardware detection (NVIDIA CUDA, AMD ROCm, Intel XPU, CPU)
- ✅ OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`)
- ✅ Cloudflare Tunnel for zero-config public HTTPS access
- ✅ Docker Compose profiles per backend
- ✅ Performance-tuned defaults (FlashInfer, prefix caching, chunked prefill)

---

## 🚀 Quick Start

### 1. Clone & Configure
```bash
git clone https://github.com/yourname/vllm-server
cd vllm-server
cp .env.example .env
# Edit .env: set MODEL_ID, HF_TOKEN, CF_TUNNEL_TOKEN
```

### 2. Detect Your Hardware
```bash
bash scripts/detect_hardware.sh
```

### 3. Start (Docker — pick your profile)
```bash
# NVIDIA GPU
docker compose --profile cuda up -d

# AMD ROCm
docker compose --profile rocm up -d

# CPU only
docker compose --profile cpu up -d
```

### 4. Native Python Launch (no Docker)
```bash
pip install vllm  # or: uv pip install vllm
bash scripts/install_cloudflared.sh
python src/launcher.py --model microsoft/Phi-3.5-mini-instruct
```

---

## 🌐 Cloudflare Tunnel (Dashboard-Managed)

1. Go to [Cloudflare Zero Trust](https://one.dash.cloudflare.com/) → **Networks → Tunnels**
2. Create a new tunnel → **Cloudflared** connector
3. Copy the **tunnel token** → paste into `.env` as `CF_TUNNEL_TOKEN`
4. Add a **Public Hostname**:
   - Subdomain: `llm`
   - Domain: `yourdomain.com`
   - Service: `http://localhost:8000`
5. The tunnel connector in Docker Compose auto-connects on startup.

---

## 🔌 API Usage

```bash
# Chat completion
curl https://llm.yourdomain.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key-here" \
  -d '{
    "model": "Phi-3.5-mini-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 200
  }'

# List loaded models
curl https://llm.yourdomain.com/v1/models
```

---

## ⚡ Performance Tuning Notes

| Feature | Effect |
|---|---|
| `--enable-prefix-caching` | Reuse KV cache for repeated prompts |
| `--enable-chunked-prefill` | Better batching for mixed workloads |
| `--gpu-memory-utilization 0.90` | Use 90% of VRAM for KV cache |
| `VLLM_ATTENTION_BACKEND=FLASHINFER` | Fastest attention on NVIDIA |
| `--max-num-batched-tokens 16384` | Higher throughput per step |
| `--tensor-parallel-size N` | Spread across N GPUs |

---

## 📦 Recommended Models by Hardware

| Hardware | Recommended Model |
|---|---|
| 8GB VRAM GPU | `Phi-3.5-mini-instruct`, `Qwen2.5-7B-Instruct` |
| 16–24GB VRAM | `Llama-3.1-8B-Instruct`, `Mistral-7B-v0.3` |
| 40–80GB VRAM | `Llama-3.1-70B-Instruct` (with TP) |
| CPU (16+ cores) | `Phi-3.5-mini-instruct`, `Qwen2-1.5B` |
| Apple Silicon M2+ | `Phi-3.5-mini-instruct` via CPU backend |
