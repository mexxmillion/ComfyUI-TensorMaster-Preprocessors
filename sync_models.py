"""Download only pinned inference files; verify every file before use."""
import argparse
import hashlib
import json
from pathlib import Path


def sync(models_dir):
    from huggingface_hub import hf_hub_download
    manifest = json.loads(Path(__file__).with_name("models.lock.json").read_text())
    for item in manifest["artifacts"]:
        target = Path(models_dir) / item["local_dir"] / item["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            hf_hub_download(item["repo"], item["file"], revision=item["revision"], local_dir=target.parent)
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if target.stat().st_size != item["size"] or digest != item["sha256"]:
            raise RuntimeError(f"Checkpoint verification failed: {target}")
        print(f"Verified {target.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True, help="ComfyUI models directory")
    sync(parser.parse_args().models_dir)
