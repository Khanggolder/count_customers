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


def test_equal_net_does_not_hide_in_out_errors() -> None:
    prediction = [
        {"direction": "IN", "crossing_direction": "A_TO_B", "occupancy_delta": "1", "timestamp_s": "1.0"},
        {"direction": "OUT", "crossing_direction": "B_TO_A", "occupancy_delta": "-1", "timestamp_s": "2.0"},
    ]
    labels = [{"direction": "IN", "timestamp_s": "1.0"}]
    report = evaluate(prediction, labels)
    assert report["mae_in"] == 0
    assert report["mae_out"] == 1
    assert report["net_mae"] == 1
    assert report["precision"] == 0.5
    assert report["recall"] == 1


def test_zero_net_mae_can_coexist_with_large_in_out_error() -> None:
    prediction = ([{"direction": "IN"}] * 10) + ([{"direction": "OUT"}] * 10)
    labels = ([{"direction": "IN"}] * 5) + ([{"direction": "OUT"}] * 5)
    report = evaluate(prediction, labels)
    assert report["net_mae"] == 0
    assert report["mae_in"] == report["mae_out"] == 5
    assert report["count_accuracy_in"] == report["count_accuracy_out"] == 0


def test_ambiguous_legacy_negative_delta_is_not_scored() -> None:
    report = evaluate([{"direction": "IN", "delta": "-1"}], [{"direction": "OUT"}])
    assert report["mae"] is None
    assert report["model_validation"] == "LEGACY_AMBIGUOUS_DIRECTION_NA"
