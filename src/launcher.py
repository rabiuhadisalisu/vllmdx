"""
vLLM Launcher — reads config, detects hardware, starts vLLM server.
Usage: python launcher.py --model <hf_model_id_or_path>
"""

import argparse
import os
import subprocess
import sys
from hardware_detect import detect_hardware, Backend

VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_HOST = os.getenv("VLLM_HOST", "0.0.0.0")


def build_vllm_command(model: str, hw, extra_args: list[str]) -> list[str]:
    """Build the vllm serve command with hardware-tuned flags."""
    cmd = [
        "vllm", "serve", model,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--dtype", hw.dtype,
        "--served-model-name", model.split("/")[-1],  # short name for API
        "--trust-remote-code",
        "--max-model-len", os.getenv("MAX_MODEL_LEN", "8192"),
        "--uvicorn-log-level", "warning",
    ]

    # Append hardware-specific optimizations
    cmd.extend(hw.extra_args)

    # User extra args override
    cmd.extend(extra_args)

    return cmd


def main():
    parser = argparse.ArgumentParser(description="vLLM Auto-Launch")
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_ID", "microsoft/Phi-3.5-mini-instruct"),
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--extra-args",
        nargs="*",
        default=[],
        help="Additional vllm serve arguments",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  vLLM Auto-Launch Server")
    print("=" * 50)

    # Detect hardware
    hw = detect_hardware()

    # Apply env vars
    for k, v in hw.env_vars.items():
        os.environ[k] = v
        print(f"  ENV: {k}={v}")

    # Build command
    cmd = build_vllm_command(args.model, hw, args.extra_args)
    print(f"\n[→] Starting vLLM: {' '.join(cmd)}\n")

    # Run (blocking)
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[!] Shutting down.")
    except subprocess.CalledProcessError as e:
        print(f"[✗] vLLM exited with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
