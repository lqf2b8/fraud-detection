import numpy as np
import pandas as pd
import pytest

from fraud_detection import preprocess

D = 86_400


def _cat(values, cats):
    return pd.Categorical(values, categories=cats)


def _frames():
    cats = ["common", "rare", "other"]
    tr = pd.DataFrame({
        "dev": _cat(["common"] * 9_990 + ["rare"] * 5 + ["other"] * 5, cats),
        "card1": [1] * 5_000 + [2] * 5_000,
        "D1": np.arange(10_000, dtype=float),
    })
    va = pd.DataFrame({
        "dev": _cat(["common", "rare", None], cats),
        "card1": [1, 3, 2],
        "D1": [1.0, 2.0, np.nan],
    })
    return tr, va


def test_catnum_groups_rare_values_and_keeps_missing_distinct():
    tr, va = _frames()
    _, va2 = preprocess.apply("catnum", tr, va, ["dev"], np.zeros(len(tr)), np.zeros(len(va)))
    assert va2["dev"].tolist() == [0.0, -2.0, -1.0]  # common, rare (<0.1%), missing


def test_freq_is_frozen_from_training_and_unseen_is_zero():
    tr, va = _frames()
    _, va2 = preprocess.apply("freq", tr, va, ["dev"], np.zeros(len(tr)), np.zeros(len(va)))
    assert va2["card1_freq"].tolist() == [0.5, 0.0, 0.5]
    assert va2["dev_freq"].iloc[0] == pytest.approx(0.999)


def test_dnorm_is_past_only():
    tr = pd.DataFrame({"D1": [10.0, 20.0, 30.0]})
    va = pd.DataFrame({"D1": [40.0]})
    t_tr, t_va = np.array([0, D, 2 * D]), np.array([3 * D])
    tr2, va2 = preprocess.apply("dnorm", tr, va, [], t_tr, t_va)
    assert np.isnan(tr2["D1"].iloc[0])                # nothing earlier to average
    assert tr2["D1"].tolist()[1:] == [10.0, 15.0]     # 20 - 10, 30 - mean(10, 20)
    assert va2["D1"].iloc[0] == 20.0                  # 40 - mean(10, 20, 30)


def test_unknown_variant_rejected():
    with pytest.raises(SystemExit):
        preprocess.parse("catnum+typo")
