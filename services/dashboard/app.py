"""Plotly Dash frontend. Talks to the model only through the API service."""

import os

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, dcc, html

API_URL = os.getenv("API_URL", "http://localhost:8000")

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Fraud Detection")
server = app.server  # for gunicorn


def get_model_info() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=5)
        return r.json()["metadata"] if r.ok else None
    except requests.RequestException:
        return None


def metric_card(label: str, value: str) -> dbc.Col:
    return dbc.Col(
        dbc.Card(dbc.CardBody([html.Small(label, className="text-muted"), html.H4(value)])),
        md=2,
    )


def field(label: str, id_: str, **kw) -> dbc.Col:
    return dbc.Col([dbc.Label(label), dbc.Input(id=id_, **kw)], md=4)


def dropdown(label: str, id_: str, options: list[str], value: str) -> dbc.Col:
    return dbc.Col([dbc.Label(label), dcc.Dropdown(options, value, id=id_, clearable=True)], md=4)


score_tab = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    field("Transaction amount ($)", "amt", type="number", value=249.95),
                    dropdown("Product code", "product", ["W", "C", "R", "H", "S"], "W"),
                    dropdown("Card network", "card4", ["visa", "mastercard", "american express", "discover"], "visa"),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dropdown("Card type", "card6", ["debit", "credit"], "debit"),
                    dropdown(
                        "Purchaser email domain",
                        "email",
                        ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com"],
                        "anonymous.com",
                    ),
                    dropdown("Device type", "device", ["desktop", "mobile"], "mobile"),
                ],
                className="mb-3",
            ),
            dbc.Button("Score transaction", id="score-btn", color="primary"),
            html.Div(id="score-out", className="mt-4"),
        ]
    ),
    className="mt-3",
)

app.layout = dbc.Container(
    [
        html.H2("Transaction Fraud Detection", className="mt-4"),
        html.P("Score transactions and review model performance.", className="text-muted"),
        dbc.Tabs(
            [
                dbc.Tab(score_tab, label="Score a transaction"),
                dbc.Tab(html.Div(id="perf-tab", className="mt-3"), label="Model performance"),
            ],
            id="tabs",
        ),
        dcc.Interval(id="load", interval=1, max_intervals=1),
    ],
    fluid=True,
)


@app.callback(
    Output("score-out", "children"),
    Input("score-btn", "n_clicks"),
    State("amt", "value"),
    State("product", "value"),
    State("card4", "value"),
    State("card6", "value"),
    State("email", "value"),
    State("device", "value"),
    prevent_initial_call=True,
)
def score(_, amt, product, card4, card6, email, device):
    txn = {
        "TransactionAmt": amt,
        "ProductCD": product,
        "card4": card4,
        "card6": card6,
        "P_emaildomain": email,
        "DeviceType": device,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json={"transactions": [txn]}, timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        return dbc.Alert(f"API error: {e}", color="danger")
    body = r.json()
    pred = body["predictions"][0]
    p = pred["fraud_probability"]
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": "%"},
            title={"text": "Fraud probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#c0392b" if pred["flagged"] else "#27ae60"},
                "threshold": {"line": {"color": "black", "width": 3}, "value": body["threshold"] * 100},
            },
        )
    ).update_layout(height=300, margin={"t": 60, "b": 20})
    verdict = dbc.Alert(
        "Flagged for review" if pred["flagged"] else "Approve",
        color="danger" if pred["flagged"] else "success",
    )
    return [verdict, dcc.Graph(figure=gauge)]


@app.callback(Output("perf-tab", "children"), Input("load", "n_intervals"), Input("tabs", "active_tab"))
def performance(*_):
    meta = get_model_info()
    if meta is None:
        return dbc.Alert("No model available yet. Train one with `make train`.", color="warning")
    m = meta["metrics"]
    cards = dbc.Row(
        [
            metric_card("ROC AUC", f"{m['roc_auc']:.3f}"),
            metric_card("PR AUC", f"{m['pr_auc']:.3f}"),
            metric_card("Precision @ top 1%", f"{m['precision_at_1pct']:.1%}"),
            metric_card("Precision @ top 5%", f"{m['precision_at_5pct']:.1%}"),
            metric_card("Base fraud rate", f"{m['base_rate']:.2%}"),
            metric_card("Threshold", f"{meta['threshold']:.3f}"),
        ],
        className="g-2 mb-3",
    )
    pr = meta["pr_curve"]
    pr_fig = go.Figure(
        go.Scatter(
            x=pr["recall"],
            y=pr["precision"],
            mode="lines",
            customdata=pr["threshold"],
            hovertemplate="recall %{x:.2f}<br>precision %{y:.2f}<br>threshold %{customdata:.3f}<extra></extra>",
        )
    ).update_layout(title="Precision–recall (validation, time-based split)",
                    xaxis_title="Recall", yaxis_title="Precision", height=420)
    fi = meta["feature_importance"]
    fi_fig = go.Figure(go.Bar(x=list(fi.values())[::-1], y=list(fi.keys())[::-1], orientation="h"))
    fi_fig.update_layout(title="Top features (gain)", height=420, margin={"l": 120})
    caption = html.Small(
        f"{meta['model_type']} trained {meta['trained_at']} on {meta['data_source']} data "
        f"({meta['n_train']:,} train rows, {m['n_valid']:,} validation rows).",
        className="text-muted",
    )
    return [cards, dbc.Row([dbc.Col(dcc.Graph(figure=pr_fig), md=6), dbc.Col(dcc.Graph(figure=fi_fig), md=6)]), caption]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
