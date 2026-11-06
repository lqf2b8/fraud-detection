import numpy as np
import pandas as pd

from fraud_detection.features import FeatureBuilder, known_label_counts

DAY = 86_400


def test_known_label_counts_respects_delay():
    # One entity: fraud on day 0, legit on day 20, query on days 10, 31 and 51; delay 30 days.
    n, f = known_label_counts(
        hist_key=["a", "a"], hist_t=[0, 20 * DAY], hist_y=[1, 0],
        q_key=["a", "a", "a", "b"], q_t=[10 * DAY, 31 * DAY, 51 * DAY, 99 * DAY], delay_secs=30 * DAY,
    )
    assert n.tolist() == [0, 1, 2, 0]  # day 10: nothing old enough; day 31: day-0 label known
    assert f.tolist() == [0, 1, 1, 0]


def test_known_label_counts_handles_missing_keys():
    n, f = known_label_counts(["a"], [0], [1], [None, "a"], [40 * DAY, 40 * DAY], 30 * DAY)
    assert np.isnan(n[0]) and n[1] == 1 and f[1] == 1


def _customer(days, labels):
    # Same customer every time: D1 = days since the first transaction.
    return pd.DataFrame(
        {
            "TransactionID": range(len(days)),
            "TransactionDT": [d * DAY for d in days],
            "TransactionAmt": 50.0,
            "card1": 1111,
            "addr1": 300.0,
            "P_emaildomain": "gmail.com",
            "D1": [d - days[0] for d in days],
            "isFraud": labels,
        }
    )


def test_training_rows_never_see_their_own_or_recent_labels():
    df = _customer([0, 5, 40], [1, 1, 0])
    X = FeatureBuilder().fit_transform(df)
    # Day 0: no history. Day 5: day-0 fraud is only 5 days old -> not yet known.
    # Day 40: both earlier frauds are now >= 30 days old.
    assert X["uid_known_fraud"].tolist() == [0, 0, 2]
    assert X["uid_known_n"].tolist() == [0, 0, 2]


def test_changing_a_label_does_not_change_its_own_features():
    a = _customer([0, 5, 40], [0, 0, 0])
    b = a.copy()
    b.loc[2, "isFraud"] = 1  # flip the last row's label
    Xa, Xb = FeatureBuilder().fit_transform(a), FeatureBuilder().fit_transform(b)
    cols = [c for c in Xa.columns if "known" in c]
    pd.testing.assert_frame_equal(Xa[cols], Xb[cols])


def test_inference_uses_training_labels_with_delay():
    fb = FeatureBuilder()
    fb.fit_transform(_customer([0, 5], [1, 0]))
    new = _customer([20, 45], [0, 0]).drop(columns="isFraud")
    new["D1"] = [20, 45]
    X = fb.transform(new)
    assert X["uid_known_fraud"].tolist() == [0, 1]  # day-0 fraud becomes known at day 30
    assert X["email_addr_known_n"].tolist() == [0, 2]
