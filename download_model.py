from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = os.getenv("MODEL_ID", "openbmb/VoxCPM2")
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DATA_DIR / "models" / "VoxCPM2")))
HF_HOME = Path(os.getenv("HF_HOME", str(DATA_DIR / "huggingface")))


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    print(f"Downloading {MODEL_ID} into persistent-volume path: {MODEL_PATH}", flush=True)
    path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_PATH),
        cache_dir=str(HF_HOME),
    )
    config = MODEL_PATH / "config.json"
    if not config.is_file():
        raise RuntimeError(f"Download did not produce expected file: {config}")
    print(f"Download finished: {path}", flush=True)


if __name__ == "__main__":
    main()
