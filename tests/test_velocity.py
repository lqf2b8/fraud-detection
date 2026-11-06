import numpy as np
import pandas as pd
import pytest

from fraud_detection import config
from fraud_detection.counting import window_sums
from fraud_detection.features import FeatureBuilder

H, D = 3_600, 86_400


@pytest.fixture(autouse=True)
def _velocity_on(monkeypatch):
    monkeypatch.setattr(config, "FEATURE_GROUPS", {"velocity"})


def test_window_sums_counts_only_the_window_before_t():
    t = [0, 30 * 60, 2 * H, 2 * D]
    n, s = window_sums(["a"] * 4, t, [10, 20, 30, 40], ["a"] * 4, t, window_secs=D)
    # At 2 days, the 1-day window starts at day 1: all three earlier rows are older.
    assert n.tolist() == [0, 1, 2, 0]
    assert s.tolist() == [0, 10, 30, 0]
    n1h, _ = window_sums(["a"] * 4, t, [1] * 4, ["a"] * 4, t, window_secs=H)
    assert n1h.tolist() == [0, 1, 0, 0]  # 2h mark: the 30-minute row is 1.5h old
    n_other, _ = window_sums(["a"], [0], [1], ["b"], [60], window_secs=D)
    assert n_other.tolist() == [0]       # other keys never count


def _txns(times, card=1111, amt=50.0):
    return pd.DataFrame({
        "TransactionID": range(len(times)), "TransactionDT": times, "TransactionAmt": amt,
        "card1": card, "addr1": 300.0, "P_emaildomain": "gmail.com", "D1": 0.0,
        "isFraud": 0,
    })


def test_training_velocity_is_past_only_and_windowed():
    X = FeatureBuilder().fit_transform(_txns([10 * D, 10 * D + 600, 10 * D + 1200, 12 * D]))
    assert X["vel_card1_n_1h"].tolist() == [0, 1, 2, 0]
    assert X["vel_card1_n_7d"].tolist() == [0, 1, 2, 3]
    assert X["vel_card1_amt_1d"].tolist() == [0, 50, 100, 0]


def test_inference_counts_training_tail_and_earlier_batch_rows():
    fb = FeatureBuilder()
    fb.fit_transform(_txns([10 * D, 10 * D + 600]))
    new = _txns([10 * D + 1800, 10 * D + 2400]).drop(columns="isFraud")
    X = fb.transform(new)
    assert X["vel_card1_n_1h"].tolist() == [2, 3]   # two from training + the earlier new one


def test_velocity_is_stable_not_cumulative():
    # Same burst pattern a month apart gives identical features (no drift with time).
    a = FeatureBuilder().fit_transform(_txns([D, D + 60, D + 120]))
    b = FeatureBuilder().fit_transform(_txns([31 * D, 31 * D + 60, 31 * D + 120]))
    cols = [c for c in a if c.startswith("vel_")]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert np.all(a["vel_card1_n_7d"].to_numpy() == [0, 1, 2])
