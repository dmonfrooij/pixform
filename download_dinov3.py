"""
download_dinov3.py - One-time download of the DINOv3 encoder for TRELLIS.2

This script downloads facebook/dinov3-vitl16-pretrain-lvd1689m and saves it
to backend/models/dinov3_vitl16/ so PIXFORM can load it fully offline afterwards.

You need to do this only ONCE. After this, TRELLIS.2 runs without any internet
access or authentication.

Requirements:
  1. Request access at: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
     (click 'Request access', takes a few minutes to be approved)
  2. Create a token at: https://huggingface.co/settings/tokens
  3. Run this script:

     Windows PowerShell:
         $env:HF_TOKEN = 'hf_your_token_here'
         .\\venv\\Scripts\\python.exe download_dinov3.py

     macOS/Linux:
         HF_TOKEN=hf_your_token_here ./venv/bin/python download_dinov3.py

After the download completes PIXFORM will find the model automatically at
backend/models/dinov3_vitl16/ - no token or internet needed anymore.
"""

import os
import sys
from pathlib import Path

REPO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
SAVE_DIR = Path(__file__).parent / "backend" / "models" / "dinov3_vitl16"

def main():
    token = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", "")).strip() or None

    if not token:
        print("ERROR: HF_TOKEN environment variable is not set.")
        print()
        print("Set it before running this script:")
        print("  Windows:  $env:HF_TOKEN = 'hf_your_token_here'")
        print("  macOS:    export HF_TOKEN=hf_your_token_here")
        print()
        print("Get a token at: https://huggingface.co/settings/tokens")
        print("Request model access at: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m")
        sys.exit(1)

    if SAVE_DIR.exists() and any(SAVE_DIR.iterdir()):
        print(f"Model already exists at {SAVE_DIR}")
        print("Delete that directory to re-download.")
        sys.exit(0)

    print(f"Downloading {REPO_ID} to {SAVE_DIR} ...")
    print("This is a ~1 GB download and happens only once.")
    print()

    try:
        from transformers import DINOv3ViTModel, DINOv3ViTConfig
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}")
        print("Run: pip install transformers huggingface_hub")
        sys.exit(1)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Download all files for the model repo
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(SAVE_DIR),
            token=token,
        )
        print()
        print(f"SUCCESS: Model saved to {SAVE_DIR}")
        print("You can now start PIXFORM without any HF_TOKEN - TRELLIS.2 will load from local disk.")
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        print()
        print("Common causes:")
        print("  - Token is invalid or expired")
        print("  - Access request not yet approved (can take a few minutes)")
        print("  - Network issue")
        sys.exit(1)


if __name__ == "__main__":
    main()

