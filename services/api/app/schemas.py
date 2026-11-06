from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    """One transaction. Named fields are the common ones; any other IEEE-CIS column
    (V1..V339, id_01.., M1.., etc.) can be passed as an extra field. Missing columns are
    treated as missing values by the model."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "TransactionID": 3663549,
                "TransactionDT": 18403224,
                "TransactionAmt": 249.95,
                "ProductCD": "W",
                "card1": 9500,
                "card4": "visa",
                "card6": "debit",
                "addr1": 315.0,
                "P_emaildomain": "anonymous.com",
                "DeviceType": "mobile",
            }
        },
    )

    TransactionID: int | None = None
    TransactionDT: int | None = None
    TransactionAmt: float | None = None
    ProductCD: str | None = None
    card1: int | None = None
    card4: str | None = None
    card6: str | None = None
    addr1: float | None = None
    P_emaildomain: str | None = None
    DeviceType: str | None = None


class PredictRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=10_000)


class Prediction(BaseModel):
    TransactionID: int | None
    fraud_probability: float
    flagged: bool
    reasons: list["Reason"] | None = None


class Reason(BaseModel):
    """One SHAP contribution. Positive pushes toward fraud (log-odds units)."""

    feature: str
    description: str
    value: str | float | None
    contribution: float


class PredictResponse(BaseModel):
    threshold: float
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    metadata: dict[str, Any]


class ExampleSummary(BaseModel):
    TransactionID: int
    amount: float | None
    product: str | None
    is_fraud: bool
    outcome: str
    score: float


class ExampleDetail(BaseModel):
    prediction: Prediction
    threshold: float
    is_fraud: bool
    outcome: str
    details: dict[str, Any]


Prediction.model_rebuild()
