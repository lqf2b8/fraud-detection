import numpy as np
import pandas as pd

from fraud_detection.features import FeatureBuilder, make_uid


def _txns():
    # Same customer (card1/addr1/email, D1 consistent with day) transacting on days 10, 11, 13.
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [10 * 86_400, 11 * 86_400, 13 * 86_400, 12 * 86_400],
            "TransactionAmt": [100.0, 50.0, 300.0, 20.0],
            "card1": [1111, 1111, 1111, 2222],
            "addr1": [300.0, 300.0, 300.0, 400.0],
            "P_emaildomain": ["gmail.com"] * 3 + ["yahoo.com"],
            "D1": [0.0, 1.0, 3.0, 0.0],
            "isFraud": [0, 0, 1, 0],
        }
    )


def test_uid_is_stable_across_a_customers_transactions():
    uid = make_uid(_txns())
    assert uid.iloc[0] == uid.iloc[1] == uid.iloc[2]
    assert uid.iloc[3] != uid.iloc[0]


def test_uid_missing_without_d1():
    df = _txns()
    df.loc[0, "D1"] = np.nan
    assert pd.isna(make_uid(df).iloc[0])


def test_training_features_are_past_only():
    X = FeatureBuilder().fit_transform(_txns())
    assert X["uid_n_prev"].tolist() == [0, 1, 2, 0]
    assert np.isnan(X["uid_amt_mean_prev"].iloc[0])
    assert X["uid_amt_mean_prev"].iloc[2] == 75.0  # mean of the two earlier amounts
    assert X["uid_amt_ratio"].iloc[2] == 4.0
    assert X["uid_secs_since_prev"].iloc[2] == 2 * 86_400


def test_training_features_ignore_future_rows():
    df = _txns()
    full = FeatureBuilder().fit_transform(df)
    first_two = FeatureBuilder().fit_transform(df.iloc[:2])
    ent = [c for c in full.columns if c.startswith(("uid_", "card1_"))]
    pd.testing.assert_frame_equal(full.iloc[:2][ent], first_two[ent], check_dtype=False)


def test_inference_uses_training_history():
    fb = FeatureBuilder()
    fb.fit_transform(_txns())
    new = pd.DataFrame(
        {"TransactionDT": [14 * 86_400], "TransactionAmt": [150.0], "card1": [1111],
         "addr1": [300.0], "P_emaildomain": ["gmail.com"], "D1": [4.0]}
    )
    X = fb.transform(new)
    assert X["uid_n_prev"].iloc[0] == 3
    assert X["uid_amt_mean_prev"].iloc[0] == 150.0
    assert X["uid_secs_since_prev"].iloc[0] == 86_400
