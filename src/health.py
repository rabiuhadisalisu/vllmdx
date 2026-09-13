"""
Simple health check + model info endpoint wrapper.
Run alongside vLLM for extra observability.
"""

import httpx
import asyncio
from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="vLLM Health Monitor")
VLLM_BASE = f"http://localhost:{os.getenv('VLLM_PORT', 8000)}"


@app.get("/status")
async def status():
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{VLLM_BASE}/health", timeout=5)
            models = await client.get(f"{VLLM_BASE}/v1/models", timeout=5)
            return {
                "vllm_healthy": r.status_code == 200,
                "models": models.json().get("data", []),
                "endpoint": VLLM_BASE,
            }
        except Exception as e:
            return {"vllm_healthy": False, "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
