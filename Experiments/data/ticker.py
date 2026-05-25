import numpy as np
import pandas as pd
import yfinance as yf


class StockHistory(yf.Ticker):
    def __init__(self, ticker_symbol="^NSEI", *args, **kwargs):
        super().__init__(ticker_symbol, *args, **kwargs)

    def get_history(self, period="5y", interval="1d", **kwargs):
        df = self.history(
            period=period,
            interval=interval,
            auto_adjust=False,
            **kwargs,
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns.name = None

        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"])
        if getattr(df["date"].dt, "tz", None) is not None:
            df["date"] = df["date"].dt.tz_localize(None)

        price_cols = ["Open", "High", "Low", "Close"]
        df[price_cols] = df[price_cols].astype(float).round(2)

        df["log_returns_close"] = np.log(df["Close"]).diff().round(6)

        df = df[["date", *price_cols, "log_returns_close"]]
        df = df.dropna(subset=[*price_cols, "log_returns_close"])

        return df.reset_index(drop=True)
