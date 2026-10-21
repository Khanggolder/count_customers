from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    packages = [
        "cv2",
        "numpy",
        "torch",
        "ultralytics",
        "fastapi",
        "sqlalchemy",
        "streamlit",
        "scipy",
    ]
    report: dict[str, object] = {
        "python": sys.version,
        "executable": sys.executable,
        "packages": {name: importlib.util.find_spec(name) is not None for name in packages},
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "nvidia_smi": shutil.which("nvidia-smi"),
    }
    if report["packages"]["torch"]:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
    if report["nvidia_smi"]:
        result = subprocess.run(
            [
                str(report["nvidia_smi"]),
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        report["gpu_driver"] = result.stdout.strip()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    required = ["cv2", "numpy", "torch", "ultralytics", "fastapi", "sqlalchemy", "streamlit"]
    return 0 if all(report["packages"][item] for item in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
