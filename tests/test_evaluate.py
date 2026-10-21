from scripts.evaluate import evaluate


def test_no_ground_truth_is_na() -> None:
    report = evaluate([{"direction": "IN", "timestamp_s": "1.0"}], None)
    assert report["mae"] is None
    assert report["precision"] is None
    assert report["recall"] is None


def test_manual_timed_labels_count_and_event_metrics() -> None:
    prediction = [
        {"direction": "IN", "timestamp_s": "1.1"},
        {"direction": "IN", "timestamp_s": "4.0"},
        {"direction": "OUT", "timestamp_s": "5.0"},
    ]
    labels = [
        {"direction": "IN", "timestamp_s": "1.0"},
        {"direction": "OUT", "timestamp_s": "5.2"},
    ]
    report = evaluate(prediction, labels, tolerance_s=0.3)
    assert report["mae"] == 0.5
    assert report["matched_events"] == 2
    assert report["precision"] == 2 / 3
    assert report["recall"] == 1.0


def test_count_only_labels_do_not_claim_precision_recall() -> None:
    report = evaluate([{"direction": "IN"}], [{"in": "2", "out": "0"}])
    assert report["mae"] == 0.5
    assert report["precision"] is None
    assert report["recall"] is None
