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
    assert {"metrics", "pr_curve", "feature_importance", "threshold"} <= meta.keys()
