import numpy as np
import pandas as pd

from fraud_detection.data import make_synthetic
from fraud_detection.explain import contributions
from fraud_detection.features import FeatureBuilder
from fraud_detection.preprocess import CatNumEncoder
from fraud_detection.routing import RoutedModel
from fraud_detection.train import fit_model, time_split


def _setup():
    df = make_synthetic(6_000)
    rep = df.sample(1_500, random_state=0).copy()       # returning customers
    rep["TransactionDT"] += 86_400 * 100
    rep["D1"] += 100
    df = pd.concat([df, rep]).sort_values("TransactionDT").reset_index(drop=True)
    tr, va = time_split(df)
    fb = FeatureBuilder()
    X_tr, X_va = fb.fit_transform(tr), fb.transform(va)
    base, _, _ = fit_model(X_tr, tr["isFraud"], 1)
    enc = CatNumEncoder().fit(X_tr, list(fb.categorical))
    new, _, _ = fit_model(enc.transform(X_tr), tr["isFraud"], 1)
    return fb, va, X_va, base, new, enc


def test_rows_are_scored_by_the_right_model():
    fb, va, X_va, base, new, enc = _setup()
    seen = fb.seen_mask(va)
    assert seen.any() and (~seen).any()
    p = RoutedModel(base, new, enc).predict_proba(X_va, seen)[:, 1]
    np.testing.assert_allclose(p[seen], base.predict_proba(X_va[seen])[:, 1])
    np.testing.assert_allclose(p[~seen], new.predict_proba(enc.transform(X_va[~seen]))[:, 1])


def test_routed_shap_sums_to_each_rows_score():
    fb, va, X_va, base, new, enc = _setup()
    seen = fb.seen_mask(va)
    m = RoutedModel(base, new, enc)
    c = m.contributions(X_va, seen)
    assert list(c.index) == list(X_va.index)
    logit = c.sum(axis=1).to_numpy()
    np.testing.assert_allclose(1 / (1 + np.exp(-logit)), m.predict_proba(X_va, seen)[:, 1], atol=1e-6)
    # Seen rows' SHAP values come from the base model.
    np.testing.assert_allclose(c[seen].to_numpy(), contributions(base, X_va[seen]).to_numpy())


def test_encoder_is_frozen_from_training():
    *_, X_va, _base, _new, enc = _setup()
    a, b = enc.transform(X_va), enc.transform(X_va)
    pd.testing.assert_frame_equal(a, b)
    assert all(str(a[c].dtype) == "float64" for c in enc.rare)
