"""Velocity features: how busy a card, address, email + address or device has been recently.

For each transaction at time t and each entity, count the entity's transactions in the last
1 hour, 1 day and 7 days (strictly before t), plus the dollars in the last day. Unlike cumulative
counts ("customers ever seen on this device"), these windows mean the same thing in any month, so
they don't drift with time. They need no labels, and they work for brand-new customers: a first-time
customer's card or device may still be in a burst.

At inference, the window sees the last 7 days of training transactions plus earlier transactions in
the same batch. In production this would be a streaming feature store.
"""

from __future__ import annotations

import pandas as pd

from fraud_detection import config
from fraud_detection.counting import window_sums

HOUR, DAY = 3_600, 86_400
WINDOWS = {"1h": HOUR, "1d": DAY, "7d": 7 * DAY}
AMOUNT_WINDOW = "1d"


def velocity_keys(df: pd.DataFrame) -> dict[str, pd.Series]:
    keys = {}
    if "card1" in df:
        keys["card1"] = df["card1"].astype("Int64").astype("string")
    if "addr1" in df:
        keys["addr1"] = df["addr1"].astype("Int64").astype("string")
    if "P_emaildomain" in df and "addr1" in df:
        keys["email_addr"] = (df["P_emaildomain"].astype("string") + "_"
                              + df["addr1"].astype("string")).where(
            df["P_emaildomain"].notna() & df["addr1"].notna())
    if "DeviceInfo" in df:
        keys["device"] = df["DeviceInfo"].astype("string")
    return keys


class VelocityTracker:
    def __init__(self) -> None:
        self.tail: pd.DataFrame | None = None  # last 7 days of training transactions

    def _events(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ["card1", "addr1", "P_emaildomain", "DeviceInfo", "TransactionAmt",
                            config.TIME_COL] if c in df]
        return df[cols]

    def fit(self, df: pd.DataFrame) -> VelocityTracker:
        if config.TIME_COL in df:
            cutoff = df[config.TIME_COL].max() - max(WINDOWS.values())
            self.tail = self._events(df[df[config.TIME_COL] >= cutoff]).reset_index(drop=True)
        return self

    def features(self, df: pd.DataFrame, history: pd.DataFrame | None) -> pd.DataFrame:
        """history = earlier transactions to count (training: none, df counts itself)."""
        if config.TIME_COL not in df:
            return pd.DataFrame(index=df.index)
        events = self._events(df) if history is None else pd.concat(
            [history, self._events(df)], ignore_index=True)
        ev_keys = velocity_keys(events)
        q_keys = velocity_keys(df)
        amt = events.get("TransactionAmt", pd.Series(0.0, index=events.index))
        out = {}
        for name, qk in q_keys.items():
            for w, secs in WINDOWS.items():
                n, s = window_sums(ev_keys[name], events[config.TIME_COL], amt, qk,
                                   df[config.TIME_COL], secs)
                out[f"vel_{name}_n_{w}"] = n
                if w == AMOUNT_WINDOW:
                    out[f"vel_{name}_amt_{w}"] = s
        return pd.DataFrame(out, index=df.index)
