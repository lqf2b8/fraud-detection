import numpy as np
import pandas as pd
import pytest

from fraud_detection import config
from fraud_detection.anomaly import AnomalyScorer
from fraud_detection.features import FeatureBuilder

DAY = 86_400


@pytest.fixture(autouse=True)
def _enable_optional_groups(monkeypatch):
    # Network and anomaly features are off by default; these tests exercise them explicitly.
    monkeypatch.setattr(config, "FEATURE_GROUPS", {"network", "anomaly"})


def _txn(tid, day, card1, device, fraud, d1=0.0):
    return {
        "TransactionID": tid, "TransactionDT": day * DAY, "TransactionAmt": 50.0,
        "card1": card1, "card2": 100.0, "addr1": 300.0, "P_emaildomain": f"u{card1}.com",
        "D1": d1, "DeviceInfo": device, "isFraud": fraud,
    }


def _ring():
    # Customers A and B share device "X". A commits fraud on day 0.
    # B uses device X on day 10 (A's fraud not yet known) and day 40 (now known: >30 days old).
    # C is a brand-new customer showing up on device X on day 45.
    return pd.DataFrame([
        _txn(1, 0, 1111, "X", 1),
        _txn(2, 10, 2222, "X", 0),
        _txn(3, 40, 2222, "X", 0, d1=30.0),
        _txn(4, 45, 3333, "X", 0),
    ])


def test_device_network_counts_other_customers_only():
    X = FeatureBuilder().fit_transform(_ring())
    # A has no prior company on X; B sees A; B again sees only A (not itself); C sees A and B.
    assert X["net_device_customers"].tolist() == [0, 1, 1, 2]


def test_fraud_neighbors_respect_label_delay():
    X = FeatureBuilder().fit_transform(_ring())
    # A's fraud (day 0) becomes known on day 30: invisible on day 10, visible on days 40 and 45.
    assert X["net_device_fraud_customers"].tolist() == [0, 0, 1, 1]
    assert X["net_device_fraud_share"].iloc[3] == 0.5


def test_own_fraud_is_not_a_fraud_neighbor():
    df = pd.DataFrame([_txn(1, 0, 1111, "X", 1), _txn(2, 40, 1111, "X", 0, d1=40.0)])
    X = FeatureBuilder().fit_transform(df)
    assert X["net_device_fraud_customers"].tolist() == [0, 0]  # A's own fraud doesn't count
    assert X["uid_known_fraud"].iloc[1] == 1                   # ...it's captured here instead


def test_inference_reaches_new_customers_via_shared_device():
    fb = FeatureBuilder()
    fb.fit_transform(_ring())
    new = pd.DataFrame([_txn(9, 50, 9999, "X", 0)]).drop(columns="isFraud")
    X = fb.transform(new)
    assert X["net_device_customers"].iloc[0] == 3
    assert X["net_device_fraud_customers"].iloc[0] == 1


def test_anomaly_score_flags_outliers():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"TransactionAmt": rng.lognormal(4, 0.3, 2000), "C1": rng.poisson(2, 2000)})
    s = AnomalyScorer().fit(df).score(pd.DataFrame({"TransactionAmt": [55.0, 50_000.0], "C1": [2, 400]}))
    assert s.iloc[1] > s.iloc[0]


def test_optional_groups_off_by_default(monkeypatch):
    monkeypatch.setattr(config, "FEATURE_GROUPS", set())
    X = FeatureBuilder().fit_transform(_ring())
    assert not [c for c in X if c.startswith("net_") or c == "anomaly_score"]
