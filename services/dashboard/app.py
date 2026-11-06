"""Plotly Dash frontend. Talks to the model only through the API service.

Tabs:
    Review queue       real validation transactions: score, decision, true outcome, SHAP reasons
    Threshold & cost   pick an operating point from review cost and review capacity
    Score a transaction  manual what-if form
    Model performance  metrics, seen vs. new customers, PR curve, SHAP importance
"""

import os

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, dash_table, dcc, html

API_URL = os.getenv("API_URL", "http://localhost:8000")
FRAUD, LEGIT, GREY = "#c0392b", "#27ae60", "#7f8c8d"

# Bootstrap (Bootswatch Flatly, MIT) is bundled in assets/ and auto-served by Dash; no CDN needed.
app = Dash(__name__, title="Fraud Detection", suppress_callback_exceptions=True)
server = app.server  # for gunicorn


# ----------------------------------------------------------------------------- API helpers
def api_get(path: str):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=10)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def get_meta() -> dict | None:
    body = api_get("/model-info")
    return body["metadata"] if body else None


def no_model() -> dbc.Alert:
    return dbc.Alert("No model available yet. Train one with `make train`, then `make up`.", color="warning")


# ----------------------------------------------------------------------------- components
def card(label: str, value: str, sub: str | None = None, md: int = 2) -> dbc.Col:
    body = [html.Small(label, className="text-muted"), html.H4(value, className="mb-0")]
    if sub:
        body.append(html.Small(sub, className="text-muted"))
    return dbc.Col(dbc.Card(dbc.CardBody(body)), md=md)


def gauge(p: float, threshold: float, flagged: bool) -> go.Figure:
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Fraud probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": FRAUD if flagged else LEGIT},
                "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100},
            },
        )
    ).update_layout(height=260, margin={"t": 50, "b": 10, "l": 30, "r": 30})


def reasons_chart(reasons: list[dict]) -> go.Figure:
    """SHAP contributions for one transaction: red pushes toward fraud, green toward legit."""
    rs = list(reversed(reasons))
    labels = [
        f"{r['description']} = {r['value'] if r['value'] is not None else 'missing'}" for r in rs
    ]
    vals = [r["contribution"] for r in rs]
    fig = go.Figure(
        go.Bar(
            x=vals, y=labels, orientation="h",
            marker_color=[FRAUD if v > 0 else LEGIT for v in vals],
            hovertemplate="%{y}<br>contribution %{x:+.2f} log-odds<extra></extra>",
        )
    )
    fig.update_layout(
        title="Why this score: top SHAP contributions",
        xaxis_title="← toward legit · contribution (log-odds) · toward fraud →",
        height=320, margin={"l": 10, "r": 10, "t": 50, "b": 40}, yaxis={"automargin": True},
    )
    return fig


# ----------------------------------------------------------------------------- tab: review queue
OUTCOME_COLORS = {"caught fraud": "success", "missed fraud": "danger",
                  "false alarm": "warning", "cleared legit": "secondary"}


def review_tab():
    ex = api_get("/examples")
    if ex is None:
        return no_model()
    if not ex:
        return dbc.Alert("No example transactions saved. Retrain with the latest trainer.", color="info")
    return dbc.Row(
        [
            dbc.Col(
                [
                    html.P(
                        "A sample of real validation transactions the model never trained on: "
                        "caught frauds, missed frauds, false alarms and correctly cleared ones.",
                        className="text-muted small",
                    ),
                    dcc.Dropdown(
                        ["caught fraud", "missed fraud", "false alarm", "cleared legit"],
                        None, id="outcome-filter", placeholder="Filter by outcome", clearable=True,
                        className="mb-2",
                    ),
                    dash_table.DataTable(
                        id="examples-table",
                        data=ex,
                        columns=[
                            {"name": "ID", "id": "TransactionID"},
                            {"name": "Amount", "id": "amount", "type": "numeric",
                             "format": dash_table.FormatTemplate.money(2)},
                            {"name": "Product", "id": "product"},
                            {"name": "Score", "id": "score", "type": "numeric",
                             "format": dash_table.Format.Format(precision=3,
                                                                scheme=dash_table.Format.Scheme.fixed)},
                            {"name": "Outcome", "id": "outcome"},
                        ],
                        row_selectable="single",
                        sort_action="native",
                        page_size=14,
                        style_cell={"fontSize": 13, "padding": "4px"},
                        style_data_conditional=[
                            {"if": {"filter_query": '{outcome} = "missed fraud"'}, "color": FRAUD},
                            {"if": {"filter_query": '{outcome} = "caught fraud"'}, "color": LEGIT},
                        ],
                    ),
                ],
                md=5,
            ),
            dbc.Col(html.Div(id="example-detail", children=html.P(
                "Select a transaction to see its score and the reasons behind it.",
                className="text-muted mt-5 text-center")), md=7),
        ],
        className="mt-3",
    )


@app.callback(Output("examples-table", "data"), Input("outcome-filter", "value"))
def filter_examples(outcome):
    ex = api_get("/examples") or []
    return [e for e in ex if outcome is None or e["outcome"] == outcome]


@app.callback(
    Output("example-detail", "children"),
    Input("examples-table", "selected_rows"),
    State("examples-table", "data"),
    prevent_initial_call=True,
)
def show_example(selected, data):
    if not selected:
        return html.P("Select a transaction.", className="text-muted")
    tid = data[selected[0]]["TransactionID"]
    d = api_get(f"/examples/{tid}")
    if d is None:
        return dbc.Alert("Could not load example from the API.", color="danger")
    p = d["prediction"]
    details = " · ".join(f"{k}: {v}" for k, v in d["details"].items() if v is not None)
    return [
        dbc.Row(
            [
                dbc.Col(dcc.Graph(figure=gauge(p["fraud_probability"], d["threshold"], p["flagged"]),
                                  config={"displayModeBar": False}), md=6),
                dbc.Col(
                    [
                        html.H5(f"Transaction {tid}", className="mt-3"),
                        html.P(details, className="small text-muted"),
                        dbc.Badge("Flagged for review" if p["flagged"] else "Approved",
                                  color="danger" if p["flagged"] else "success", className="me-2"),
                        dbc.Badge(f"Actually {'FRAUD' if d['is_fraud'] else 'legit'}",
                                  color="dark" if d["is_fraud"] else "light", text_color=None
                                  if d["is_fraud"] else "dark", className="me-2"),
                        dbc.Badge(d["outcome"], color=OUTCOME_COLORS.get(d["outcome"], "secondary")),
                    ],
                    md=6,
                ),
            ]
        ),
        dcc.Graph(figure=reasons_chart(p["reasons"]), config={"displayModeBar": False}),
    ]


# ----------------------------------------------------------------------------- tab: threshold & cost
def cost_tab():
    meta = get_meta()
    if meta is None:
        return no_model()
    if "threshold_table" not in meta:
        return dbc.Alert("Retrain with the latest trainer to enable cost analysis.", color="info")
    default_cost = meta["operating_point"]["review_cost_usd"]
    default_fp = meta["operating_point"].get("false_positive_cost_usd", 0)
    return html.Div(
        [
            html.P(
                "Every flagged transaction costs an analyst review, every false alarm can also "
                "cost customer friction, and every missed fraud costs its dollar amount. Set your "
                "assumptions and the review team's capacity, and the dashboard picks the cheapest "
                "threshold within that capacity.",
                className="text-muted",
            ),
            dbc.Row(
                [
                    dbc.Col([dbc.Label("Cost per manual review ($)"),
                             dcc.Slider(1, 50, 1, value=default_cost, id="review-cost",
                                        marks={i: f"${i}" for i in (1, 10, 20, 30, 40, 50)},
                                        tooltip={"placement": "bottom"})], md=4),
                    dbc.Col([dbc.Label("Customer friction per false alarm ($)"),
                             dcc.Slider(0, 100, 5, value=default_fp, id="fp-cost",
                                        marks={i: f"${i}" for i in (0, 25, 50, 75, 100)},
                                        tooltip={"placement": "bottom"})], md=4),
                    dbc.Col([dbc.Label("Review capacity (max % flagged)"),
                             dcc.Slider(0.5, 30, 0.5, value=30, id="capacity",
                                        marks={i: f"{i}%" for i in (1, 5, 10, 20, 30)},
                                        tooltip={"placement": "bottom"})], md=4),
                ],
                className="mb-3",
            ),
            html.Div(id="cost-cards"),
            dcc.Graph(id="cost-chart"),
        ],
        className="mt-3",
    )


@app.callback(
    Output("cost-cards", "children"),
    Output("cost-chart", "figure"),
    Input("review-cost", "value"),
    Input("fp-cost", "value"),
    Input("capacity", "value"),
)
def update_cost(review_cost, fp_cost, capacity):
    meta = get_meta()
    t = meta["threshold_table"]
    flag = np.array(t["flag_rate"])
    caught_amt = np.array(t["fraud_amt_caught"])
    fps = np.array(t.get("false_positives", np.array(t["n_flagged"]) - np.array(t["frauds_caught"])))
    friction = (fp_cost or 0) * fps
    cost = review_cost * np.array(t["n_flagged"]) + friction + (t["fraud_amt_total"] - caught_amt)
    allowed = flag <= capacity / 100
    i = int(np.argmin(np.where(allowed, cost, np.inf)))
    no_model_cost = t["fraud_amt_total"]
    savings = no_model_cost - cost[i]

    cards = dbc.Row(
        [
            card("Threshold", f"{t['threshold'][i]:.3f}"),
            card("Flagged", f"{flag[i]:.1%}", f"{t['n_flagged'][i]:,} reviews"),
            card("Precision", f"{t['precision'][i]:.0%}", "flags that are fraud"),
            card("Frauds caught", f"{t['recall'][i]:.0%}", f"{t['frauds_caught'][i]:,} of {t['n_fraud']:,}"),
            card("Fraud $ caught", f"{caught_amt[i] / t['fraud_amt_total']:.0%}",
                 f"${caught_amt[i]:,.0f} of ${t['fraud_amt_total']:,.0f}"),
            card("Savings vs. no model", f"${savings:,.0f}", f"{savings / no_model_cost:.0%} lower cost"),
        ],
        className="g-2 mb-2",
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=flag * 100, y=cost, name="Total cost", line={"color": FRAUD, "width": 3},
                             hovertemplate="flag %{x:.1f}%<br>cost $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=flag * 100, y=review_cost * np.array(t["n_flagged"]), name="Review cost",
                             line={"color": GREY, "dash": "dot"}))
    if fp_cost:
        fig.add_trace(go.Scatter(x=flag * 100, y=friction, name="False-alarm friction",
                                 line={"color": "#e67e22", "dash": "dashdot"}))
    fig.add_trace(go.Scatter(x=flag * 100, y=t["fraud_amt_total"] - caught_amt, name="Fraud $ missed",
                             line={"color": GREY, "dash": "dash"}))
    fig.add_trace(go.Scatter(x=[flag[i] * 100], y=[cost[i]], mode="markers", name="Chosen point",
                             marker={"size": 14, "color": "black", "symbol": "diamond"}))
    if capacity < 30:
        fig.add_vrect(x0=capacity, x1=50, fillcolor=GREY, opacity=0.12, line_width=0,
                      annotation_text="over capacity", annotation_position="top left")
    fig.update_layout(
        title="Operating cost by review budget (validation period)",
        xaxis={"title": "% of transactions flagged for review", "type": "log",
                   "tickvals": [0.1, 0.5, 1, 2, 5, 10, 20, 50]},
        yaxis_title="Cost ($)", height=430, legend={"orientation": "h", "y": -0.2},
    )
    return cards, fig


# ----------------------------------------------------------------------------- tab: manual score
def field(label, id_, **kw):
    return dbc.Col([dbc.Label(label), dbc.Input(id=id_, **kw)], md=4)


def dropdown(label, id_, options, value):
    return dbc.Col([dbc.Label(label), dcc.Dropdown(options, value, id=id_, clearable=True)], md=4)


score_tab = dbc.Card(
    dbc.CardBody(
        [
            html.P("What-if scoring from a few fields. Unspecified fields are treated as missing, "
                   "so for realistic scores use the Review queue.", className="text-muted small"),
            dbc.Row([field("Transaction amount ($)", "amt", type="number", value=249.95),
                     dropdown("Product code", "product", ["W", "C", "R", "H", "S"], "W"),
                     dropdown("Card network", "card4", ["visa", "mastercard", "american express", "discover"],
                              "visa")], className="mb-3"),
            dbc.Row([dropdown("Card type", "card6", ["debit", "credit"], "debit"),
                     dropdown("Purchaser email domain", "email",
                              ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com"],
                              "anonymous.com"),
                     dropdown("Device type", "device", ["desktop", "mobile"], "mobile")], className="mb-3"),
            dbc.Button("Score transaction", id="score-btn", color="primary"),
            html.Div(id="score-out", className="mt-4"),
        ]
    ),
    className="mt-3",
)


@app.callback(
    Output("score-out", "children"),
    Input("score-btn", "n_clicks"),
    State("amt", "value"), State("product", "value"), State("card4", "value"),
    State("card6", "value"), State("email", "value"), State("device", "value"),
    prevent_initial_call=True,
)
def score(_, amt, product, card4, card6, email, device):
    txn = {"TransactionAmt": amt, "ProductCD": product, "card4": card4, "card6": card6,
           "P_emaildomain": email, "DeviceType": device}
    try:
        r = requests.post(f"{API_URL}/predict", params={"explain": "true"},
                          json={"transactions": [txn]}, timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        return dbc.Alert(f"API error: {e}", color="danger")
    body = r.json()
    pred = body["predictions"][0]
    verdict = dbc.Alert("Flagged for review" if pred["flagged"] else "Approve",
                        color="danger" if pred["flagged"] else "success")
    return [verdict,
            dbc.Row([dbc.Col(dcc.Graph(figure=gauge(pred["fraud_probability"], body["threshold"],
                                                     pred["flagged"])), md=5),
                     dbc.Col(dcc.Graph(figure=reasons_chart(pred["reasons"])), md=7)])]


# ----------------------------------------------------------------------------- tab: performance
def performance_tab():
    meta = get_meta()
    if meta is None:
        return no_model()
    m = meta["metrics"]
    cards = dbc.Row(
        [
            card("PR AUC", f"{m['pr_auc']:.3f}", f"random = {m['base_rate']:.3f}"),
            card("ROC AUC", f"{m['roc_auc']:.3f}"),
            card("Precision @ top 1%", f"{m['precision_at_1pct']:.1%}"),
            card("Precision @ top 5%", f"{m['precision_at_5pct']:.1%}"),
            card("Fraud rate", f"{m['base_rate']:.2%}"),
            card("Feature set", meta.get("feature_set", "—")),
        ],
        className="g-2 mb-3",
    )

    seg = meta.get("segments", {})
    seg_rows = [
        html.Tr([html.Td(k.replace("_", " ")), html.Td(f"{v['share']:.1%}"), html.Td(f"{v['base_rate']:.2%}"),
                 html.Td(f"{v['pr_auc']:.3f}"), html.Td(f"{v['roc_auc']:.3f}")])
        for k, v in seg.items()
    ]
    an = meta.get("anomaly_only") or {}
    if an:
        seg_rows.append(html.Tr([html.Td(html.I("anomaly score alone, new customers")), html.Td("—"),
                                 html.Td("—"), html.Td(f"{an.get('new_customer', {}).get('pr_auc', float('nan')):.3f}"),
                                 html.Td(f"{an.get('new_customer', {}).get('roc_auc', float('nan')):.3f}")]))
    seg_table = dbc.Table(
        [html.Thead(html.Tr([html.Th(h) for h in ["Segment", "Share", "Fraud rate", "PR AUC", "ROC AUC"]])),
         html.Tbody(seg_rows)], bordered=False, size="sm", className="mb-3",
    ) if seg_rows else None

    pr = meta["pr_curve"]
    pr_fig = go.Figure(go.Scatter(
        x=pr["recall"], y=pr["precision"], mode="lines", line={"color": FRAUD}, customdata=pr["threshold"],
        hovertemplate="recall %{x:.2f}<br>precision %{y:.2f}<br>threshold %{customdata:.3f}<extra></extra>",
    )).update_layout(title="Precision–recall (validation, time-based split)", xaxis_title="Recall",
                     yaxis_title="Precision", height=460)

    imp = meta.get("shap_importance") or meta.get("feature_importance", {})
    desc = meta.get("feature_descriptions", {})
    names = [desc.get(k, k) for k in list(imp)[:20]][::-1]
    title = "Top features (mean |SHAP|)" if "shap_importance" in meta else "Top features (gain)"
    fi_fig = go.Figure(go.Bar(x=list(imp.values())[:20][::-1], y=names, orientation="h", marker_color=GREY))
    fi_fig.update_layout(title=title, height=460, margin={"l": 10}, yaxis={"automargin": True})

    caption = html.Small(
        f"{meta['model_type']} trained {meta['trained_at']} on {meta['data_source']} data "
        f"({meta['n_train']:,} train rows, {m['n_valid']:,} validation rows; "
        f"label delay {meta.get('label_delay_days', '—')} days).",
        className="text-muted",
    )
    return html.Div([cards, html.H6("Seen vs. new customers") if seg_table else None, seg_table,
                     dbc.Row([dbc.Col(dcc.Graph(figure=pr_fig), md=6), dbc.Col(dcc.Graph(figure=fi_fig), md=6)]),
                     caption], className="mt-3")


# ----------------------------------------------------------------------------- layout
app.layout = dbc.Container(
    [
        html.H2("Transaction Fraud Detection", className="mt-4"),
        html.P("Review flagged transactions, choose an operating threshold, and inspect model performance.",
               className="text-muted"),
        dbc.Tabs(
            [
                dbc.Tab(label="Review queue", tab_id="review"),
                dbc.Tab(label="Threshold & cost", tab_id="cost"),
                dbc.Tab(label="Score a transaction", tab_id="score"),
                dbc.Tab(label="Model performance", tab_id="perf"),
            ],
            id="tabs", active_tab="review",
        ),
        html.Div(id="tab-content"),
    ],
    fluid=True,
)


@app.callback(Output("tab-content", "children"), Input("tabs", "active_tab"))
def render_tab(tab):
    return {"review": review_tab, "cost": cost_tab, "score": lambda: score_tab,
            "perf": performance_tab}[tab]()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
