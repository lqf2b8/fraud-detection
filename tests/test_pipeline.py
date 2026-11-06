import pytest
from fastapi.testclient import TestClient

from fraud_detection import config
from fraud_detection.data import make_synthetic
from fraud_detection.features import FeatureBuilder


def test_feature_builder_handles_missing_columns():
    df = make_synthetic(n=500)
    fb = FeatureBuilder().fit(df)
    X = fb.transform(df[["TransactionAmt", "ProductCD"]])  # most columns absent
    assert list(X.columns) == fb.feature_names
    assert len(X) == 500


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    model_dir = tmp_path_factory.mktemp("models")
    monkeypatch_module.setattr(config, "MODEL_DIR", model_dir)
    monkeypatch_module.setattr(config, "MODEL_PATH", model_dir / "model.joblib")
    monkeypatch_module.setattr(config, "METADATA_PATH", model_dir / "metadata.json")
    monkeypatch_module.setattr(config, "EXAMPLES_PATH", model_dir / "examples.parquet")

    from fraud_detection.train import train

    meta = train(make_synthetic(n=5_000), "synthetic")
    assert meta["metrics"]["roc_auc"] > 0.6

    from app import main

    monkeypatch_module.setattr(main, "MODEL_DIR", model_dir)
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict(client):
    payload = {"transactions": [{"TransactionID": 1, "TransactionAmt": 120.0, "ProductCD": "W"},
                                {"TransactionAmt": 5000.0, "DeviceType": "mobile"}]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert all(0 <= p["fraud_probability"] <= 1 for p in preds)


def test_model_info(client):
    meta = client.get("/model-info").json()["metadata"]
    assert {"metrics", "pr_curve", "shap_importance", "threshold", "threshold_table",
            "operating_point"} <= meta.keys()


def test_predict_with_explanations(client):
    r = client.post("/predict?explain=true", json={"transactions": [{"TransactionAmt": 75.0}]})
    reasons = r.json()["predictions"][0]["reasons"]
    assert len(reasons) == 5
    assert {"feature", "description", "value", "contribution"} <= reasons[0].keys()


def test_explanations_sum_to_model_score(client):
    import numpy as np
    import pandas as pd
    from app import main

    from fraud_detection.explain import contributions

    df = pd.DataFrame([{"TransactionAmt": 75.0, "ProductCD": "C"}])
    X = main.STATE["features"].transform(df)
    contrib = contributions(main.STATE["model"], X)
    p = main.STATE["model"].predict_proba(X)[:, 1][0]
    assert np.isclose(1 / (1 + np.exp(-contrib.sum(axis=1).iloc[0])), p, atol=1e-6)


def test_examples_round_trip(client):
    ex = client.get("/examples").json()
    assert ex and {"caught fraud", "cleared legit"} <= {e["outcome"] for e in ex}
    detail = client.get(f"/examples/{ex[0]['TransactionID']}").json()
    # The API re-scores the raw example and must reproduce the trainer's score exactly.
    assert abs(detail["prediction"]["fraud_probability"] - ex[0]["score"]) < 1e-5
    assert client.get("/examples/1").status_code == 404
