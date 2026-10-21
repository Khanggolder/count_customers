import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "import_pcds.py"
spec = importlib.util.spec_from_file_location("import_pcds", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_label_parser_named_counts(tmp_path: Path) -> None:
    label = tmp_path / "label.txt"
    label.write_text("Entering: 3\nExiting: 2\n", encoding="utf-8")
    assert module.parse_label(label) == (3, 2)


def test_scan_front_back(tmp_path: Path) -> None:
    folder = tmp_path / "BUS_001_front"
    folder.mkdir()
    (folder / "label.txt").write_text("in=1 out=0", encoding="utf-8")
    rows = module.scan(tmp_path)
    assert len(rows) == 1
    assert rows[0]["camera"] == "front"


def test_documented_pcds_per_clip_rows(tmp_path: Path) -> None:
    label = tmp_path / "label.txt"
    label.write_text(
        "1.0, 0.0, 0.0\n0.0, 1.0, 0.0\n0.0, 0.0, 1.0\n0.0, 0.0, 0.0\n"
        "depth_001.avi, 2, 1, 3\ndepth_002.avi, 0, 4, 1\n",
        encoding="utf-8",
    )
    assert module.parse_pcds_rows(label) == [
        {"clip": "depth_001.avi", "entering_gt": 2, "exiting_gt": 1, "video_type": 3},
        {"clip": "depth_002.avi", "entering_gt": 0, "exiting_gt": 4, "video_type": 1},
    ]
