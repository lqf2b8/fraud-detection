import numpy as np

from fraud_detection.bagging import BaggedModel, average_logits
from fraud_detection.data import make_synthetic
from fraud_detection.explain import contributions
from fraud_detection.features import FeatureBuilder
from fraud_detection.train import fit_bagged, fit_model, time_split


def _data():
    tr, va = time_split(make_synthetic(6_000))
    fb = FeatureBuilder()
    return fb.fit_transform(tr), tr["isFraud"], fb.transform(va)


def test_bag_shap_sums_to_bag_score():
    X, y, X_va = _data()
    bag, _, trees = fit_bagged(X, y, 3, 1)
    assert isinstance(bag, BaggedModel) and len(trees) == 3
    c = contributions(bag, X_va)
    p = bag.predict_proba(X_va)[:, 1]
    np.testing.assert_allclose(1 / (1 + np.exp(-c.sum(axis=1).to_numpy())), p, atol=1e-6)


def test_bag_is_the_log_odds_average_of_its_members():
    X, y, X_va = _data()
    bag, _, _ = fit_bagged(X, y, 3, 1)
    members = [m.predict_proba(X_va)[:, 1] for m in bag.members]
    np.testing.assert_allclose(bag.predict_proba(X_va)[:, 1], average_logits(members), atol=1e-9)


def test_single_seed_is_a_plain_model():
    X, y, _ = _data()
    model, _, _ = fit_bagged(X, y, 1, 7)
    plain, _, _ = fit_model(X, y, 7)
    assert not isinstance(model, BaggedModel)
    assert model.n_estimators == plain.n_estimators
