from busreid.occupancy import OccupancyEstimator


def test_paired_occupancy_and_negative_warning() -> None:
    result = OccupancyEstimator(initial=1, paired_sources=True).estimate(1, 4)
    assert result.value == -2
    assert result.warning is not None


def test_unpaired_is_na() -> None:
    result = OccupancyEstimator(initial=10, paired_sources=False).estimate(3, 1)
    assert result.value is None
    assert "UNPAIRED" in result.warning

