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
    "pexels-bus-boarding": {
        "url": "https://videos.pexels.com/video-files/31337217/13375520_3840_2160_30fps.mp4",
        "filename": "pexels_bus_boarding_31337217.mp4",
        "page": "https://www.pexels.com/video/busy-city-bus-stop-with-passengers-boarding-31337217/",
        "license_url": "https://www.pexels.com/license/",
        "license_note": "Pexels License: free to use; source and author Shariar Tawsif retained.",
        "technical_suitability": "Detector/tracker smoke only: exterior moving view, not fixed onboard doorway camera, no count GT.",
    }
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, retries: int = 3, timeout: int = 60) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "BusStop-Flow-AI/0.2 research MVP"})
            with urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            temporary.replace(destination)
            return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"Download failed after {retries} attempts: {exc}") from exc
            time.sleep(attempt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--yes", action="store_true", help="Confirm download of the public asset")
    args = parser.parse_args()
    info = SOURCES[args.source]
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / info["filename"]
    if not args.yes:
        print(f"Will download {info['page']} to {destination}. Add --yes to continue.")
        return 2
    if not destination.is_file():
        download(info["url"], destination)
    metadata = probe_video(destination)
    if not metadata.frame_count or metadata.frame_count < 2 or metadata.width < 64 or metadata.height < 64:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Downloaded file is not a decodable video")
    manifest = {
        "source_id": args.source,
        "file": str(destination.relative_to(ROOT)),
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
        "metadata": {key: getattr(metadata, key) for key in metadata.__slots__},
        "source_page": info["page"],
        "license_url": info["license_url"],
        "license_note": info["license_note"],
        "technical_suitability": info["technical_suitability"],
        "content_review": "PENDING_MANUAL_FRAME_REVIEW",
    }
    manifest_path = destination.with_suffix(".source.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
