from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

OFFICIAL_MODEL = {
    "name": "osnet_x0_25_msmt17",
    "file_id": "1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e",
    "model_zoo": "https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO",
    "training": "MSMT17 combineall, person Re-ID",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/osnet_x0_25_msmt17.pt"))
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if not args.yes:
        print(f"Sẽ tải checkpoint Person Re-ID từ model zoo chính thức vào {args.output}; thêm --yes để xác nhận.")
        return 2
    import gdown
    import torch

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=OFFICIAL_MODEL["file_id"], output=str(args.output), quiet=False)
    if not result or not args.output.is_file():
        raise RuntimeError("Không tải được checkpoint chính thức")
    checkpoint = torch.load(args.output, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint không có cấu trúc state dictionary")
    manifest = {
        **OFFICIAL_MODEL,
        "file": str(args.output.resolve()),
        "bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "validated": True,
        "state_tensor_count": len(checkpoint),
        "key_sample": sorted(checkpoint.keys())[:8],
    }
    args.output.with_suffix(".source.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

