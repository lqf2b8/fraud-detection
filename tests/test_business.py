import numpy as np

from fraud_detection.evaluate import cost_optimal_point, operating_cost, threshold_table
from fraud_detection.explain import describe


def _table():
    rng = np.random.default_rng(0)
    y = (rng.random(10_000) < 0.04).astype(int)
    scores = np.clip(y * 0.5 + rng.random(10_000) * 0.6, 0, 1)
    amounts = rng.lognormal(4, 1, 10_000)
    return threshold_table(y, scores, amounts), y, amounts


def test_threshold_table_is_consistent():
    t, y, amounts = _table()
    assert t["n_fraud"] == y.sum()
    assert np.isclose(t["fraud_amt_total"], (y * amounts).sum(), atol=0.01)
    assert np.all(np.diff(t["recall"]) >= 0)            # flagging more never catches fewer
    assert np.all(np.diff(t["threshold"]) <= 0)         # thresholds fall as budget grows
    assert t["recall"][-1] <= 1.0


def test_cost_optimum_moves_with_review_cost():
    t, *_ = _table()
    cheap, pricey = cost_optimal_point(t, 1.0), cost_optimal_point(t, 200.0)
    assert cheap["flag_rate"] >= pricey["flag_rate"]   # expensive reviews -> flag less
    assert cheap["cost_usd"] == min(operating_cost(t, 1.0)).round(2)
    assert cheap["cost_usd"] < cheap["cost_no_model_usd"]


def test_describe():
    assert describe("uid_known_fraud") == "Confirmed prior frauds on this customer"
    assert describe("email_addr_known_fraud_rate") == "Confirmed fraud rate on this email + address"
    assert describe("V258") == "Vesta engineered feature V258"
    assert describe("card1_amt_ratio") == "Amount vs. this card's typical amount"
