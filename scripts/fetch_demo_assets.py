from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from busreid.video import probe_video  # noqa: E402

SOURCES = {
    "github-bus-counter": {
        "url": "https://raw.githubusercontent.com/sanjarbek1030/BusOccupancyCounter/main/bus_input.mp4",
        "filename": "bus_occupancy_counter_input.mp4",
        "page": "https://github.com/sanjarbek1030/BusOccupancyCounter",
        "license_note": "Repository README states MIT, but the clip visibly contains a Shutterstock watermark; separate video rights are unconfirmed. Internal research verification only, not public redistribution.",
    },
    "pexels-bus-boarding": {
        "url": "https://videos.pexels.com/video-files/31337217/13375520_3840_2160_30fps.mp4",
        "filename": "pexels_bus_boarding_31337217.mp4",
        "page": "https://www.pexels.com/video/busy-city-bus-stop-with-passengers-boarding-31337217/",
        "license_note": "Pexels page marks the clip Free to use under the Pexels License; attribution retained in DATA_SOURCES.md.",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, retries: int = 3, timeout: int = 30) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "BusReID-AI/0.1 research MVP"})
            with urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                while block := response.read(1024 * 1024):
                    output.write(block)
                    downloaded += len(block)
                    if total:
                        print(f"\r{downloaded / total:6.1%} ({downloaded / 1_048_576:.1f} MiB)", end="")
            print()
            temporary.replace(destination)
            return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"Tải thất bại sau {retries} lần: {exc}") from exc
            time.sleep(attempt)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--yes", action="store_true", help="Xác nhận tải asset công khai")
    args = parser.parse_args()
    info = SOURCES[args.source]
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / str(info["filename"])
    if not args.yes:
        print(f"Sẽ tải {info['url']} vào {destination}. Thêm --yes để xác nhận.")
        return 2
    download(str(info["url"]), destination)
    metadata = probe_video(destination)
    if metadata.frame_count is None or metadata.frame_count < 2 or metadata.width < 64 or metadata.height < 64:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Tệp tải xuống không decode thành video hợp lệ")
    manifest = {
        "source_id": args.source,
        "file": str(destination),
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
        "metadata": {key: getattr(metadata, key) for key in metadata.__slots__},
        "source_page": info["page"],
        "license_note": info["license_note"],
        "content_review": "PENDING_MANUAL_FRAME_REVIEW",
    }
    manifest_path = destination.with_suffix(".source.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

