import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADE_PATTERN = re.compile(
    r"^\s*(BUY|SELL)\s+(CE|PE)\s+([0-9]+(?:\.[0-9]+)?)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Trade:
    action: str
    side: str
    strike: float
    premium: float

    @property
    def sign(self):
        return 1.0 if self.action == "BUY" else -1.0

    @property
    def label(self):
        strike = int(self.strike) if self.strike.is_integer() else self.strike
        return f"{self.action} {self.side} {strike}"


class TradeBuilder:
    """
    Build trades and evaluate their final-day P&L distribution.

    Trade format is case-insensitive:
        BUY CE 25000
        sell pe 24500

    P&L is per option unit, not lot-adjusted.
    """

    REQUIRED_COLUMNS = ["expiryDate", "days_to_expiry", "strikePrice", "lastPrice"]

    def __init__(self, CE, PE):
        self.CE = self._normalize_chain(CE, "CE")
        self.PE = self._normalize_chain(PE, "PE")
        self.expiry = self._extract_expiry()
        self.days_to_expiry = self._extract_days_to_expiry()

    def validate_trade(self, trade):
        """
        Validate and normalize a trade string.

        Returns a tuple: (ACTION, SIDE, STRIKE).
        """
        if not isinstance(trade, str):
            raise TypeError("Trade must be a string like 'BUY CE 25000'.")

        match = TRADE_PATTERN.match(trade.upper())
        if match is None:
            raise ValueError(
                "Invalid trade format. Expected: 'BUY CE 25000' or 'SELL PE 25000'."
            )

        action, side, strike = match.groups()
        return action, side, float(strike)

    def build_trades(self, trades):
        """Convert trade strings into Trade objects with market premiums attached."""
        return [self._parse_trade(trade) for trade in trades]

    def payoff(
        self,
        trades,
        final_prices=None,
        mean=None,
        std=None,
        n_samples=20_000,
        random_state=None,
        LOT_SIZE=65,
    ):
        """
        Return final-day P&L distribution for the given trades.

        Parameters
        ----------
        trades : list[str] or list[Trade]
            Trades in 'BUY/SELL CE/PE STRIKE' format.
        final_prices : array-like, optional
            Explicit final-day underlying prices. Use this for your own sampled
            distribution, implied distribution samples, or scenario paths.
        mean, std : float, optional
            If final_prices is not supplied, sample final-day prices from a
            normal distribution with this mean and standard deviation.
        n_samples : int
            Number of normal samples to generate when mean/std are supplied.
        LOT_SIZE : int or float
            Contract lot size. P&L is returned per lot.

        Returns
        -------
        dict
            Contains expiry metadata, final prices, total P&L samples,
            individual leg P&L matrix, and simple distribution statistics.
        """
        built_trades = self._ensure_trades(trades)
        prices = self._final_prices(
            final_prices=final_prices,
            mean=mean,
            std=std,
            n_samples=n_samples,
            random_state=random_state,
        )
        leg_pnl = self._legs_pnl(built_trades, prices) * float(LOT_SIZE)
        pnl = leg_pnl.sum(axis=0)

        return {
            "expiry": self.expiry,
            "days_to_expiry": self.days_to_expiry,
            "LOT_SIZE": LOT_SIZE,
            "final_prices": prices,
            "pnl": pnl,
            "leg_pnl": leg_pnl,
            "trades": [trade.label for trade in built_trades],
            "summary": self._summary(pnl),
        }

    def payoff_df(
        self,
        trades,
        final_prices=None,
        mean=None,
        std=None,
        n_samples=20_000,
        random_state=None,
        LOT_SIZE=65,
    ):
        """Return final-day P&L samples as a DataFrame for plotting."""
        result = self.payoff(
            trades=trades,
            final_prices=final_prices,
            mean=mean,
            std=std,
            n_samples=n_samples,
            random_state=random_state,
            LOT_SIZE=LOT_SIZE,
        )
        df = pd.DataFrame({"final_price": result["final_prices"]})
        for label, values in zip(result["trades"], result["leg_pnl"]):
            df[label] = values
        df["pnl"] = result["pnl"]
        return df

    def payoff_curve(self, trades, price_grid, LOT_SIZE=65):
        """
        Evaluate P&L over a deterministic price grid for payoff diagrams.

        This does not create a probability distribution. Use payoff(...) with
        final_prices or mean/std for sampled final-day P&L.
        """
        built_trades = self._ensure_trades(trades)
        prices = self._as_price_array(price_grid, name="price_grid")
        leg_pnl = self._legs_pnl(built_trades, prices) * float(LOT_SIZE)
        return {
            "expiry": self.expiry,
            "days_to_expiry": self.days_to_expiry,
            "LOT_SIZE": LOT_SIZE,
            "price_grid": prices,
            "pnl": leg_pnl.sum(axis=0),
            "leg_pnl": leg_pnl,
            "trades": [trade.label for trade in built_trades],
        }

    def _parse_trade(self, trade):
        action, side, strike = self.validate_trade(trade)
        premium = self._premium(side, strike)
        return Trade(action=action, side=side, strike=strike, premium=premium)

    def _ensure_trades(self, trades):
        if not isinstance(trades, (list, tuple)) or len(trades) == 0:
            raise ValueError("trades must be a non-empty list.")
        if all(isinstance(trade, Trade) for trade in trades):
            return list(trades)
        return self.build_trades(trades)

    def _legs_pnl(self, trades, prices):
        strikes = np.array([trade.strike for trade in trades], dtype=float)[:, None]
        signs = np.array([trade.sign for trade in trades], dtype=float)[:, None]
        premiums = np.array([trade.premium for trade in trades], dtype=float)[:, None]
        sides = np.array([trade.side for trade in trades])[:, None]
        final_prices = prices[None, :]

        ce_intrinsic = np.maximum(final_prices - strikes, 0.0)
        pe_intrinsic = np.maximum(strikes - final_prices, 0.0)
        intrinsic = np.where(sides == "CE", ce_intrinsic, pe_intrinsic)
        return signs * (intrinsic - premiums)

    def _premium(self, side, strike):
        chain = self.CE if side == "CE" else self.PE
        rows = chain[np.isclose(chain["strikePrice"], strike)]
        if rows.empty:
            available = chain["strikePrice"].sort_values().to_numpy()
            raise ValueError(
                f"{side} strike {strike:g} not found. "
                f"Available strike range: {available[0]:g} to {available[-1]:g}."
            )
        return float(rows["lastPrice"].iloc[0])

    def _final_prices(self, final_prices, mean, std, n_samples, random_state):
        if final_prices is not None:
            return self._as_price_array(final_prices, name="final_prices")

        if mean is None or std is None:
            raise ValueError(
                "Provide either final_prices or both mean and std for normal sampling."
            )

        if std <= 0:
            raise ValueError("std must be positive.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        rng = np.random.default_rng(random_state)
        samples = rng.normal(loc=float(mean), scale=float(std), size=int(n_samples))
        return samples[samples > 0]

    @staticmethod
    def _as_price_array(values, name):
        prices = np.asarray(values, dtype=float)
        if prices.ndim != 1 or len(prices) == 0:
            raise ValueError(f"{name} must be a non-empty 1D array.")
        if not np.all(np.isfinite(prices)):
            raise ValueError(f"{name} must contain only finite values.")
        return prices

    @staticmethod
    def _summary(pnl):
        return {
            "n_samples": int(len(pnl)),
            "prob_profit": float(np.mean(pnl > 0)),
            "expected_pnl": float(np.mean(pnl)),
            "median_pnl": float(np.median(pnl)),
            "min_pnl": float(np.min(pnl)),
            "max_pnl": float(np.max(pnl)),
            "p05": float(np.percentile(pnl, 5)),
            "p95": float(np.percentile(pnl, 95)),
        }

    def _extract_expiry(self):
        ce_expiry = self._chain_scalar(self.CE, "expiryDate")
        pe_expiry = self._chain_scalar(self.PE, "expiryDate")
        if ce_expiry != pe_expiry:
            raise ValueError(f"CE and PE expiry mismatch: {ce_expiry} vs {pe_expiry}.")
        return ce_expiry

    def _extract_days_to_expiry(self):
        ce_days = self._chain_scalar(self.CE, "days_to_expiry")
        pe_days = self._chain_scalar(self.PE, "days_to_expiry")
        ce_days = float(ce_days)
        pe_days = float(pe_days)
        if not np.isclose(ce_days, pe_days, atol=0.5, equal_nan=False):
            raise ValueError(
                f"CE and PE days_to_expiry mismatch: {ce_days} vs {pe_days}."
            )
        return float(np.mean([ce_days, pe_days]))

    @classmethod
    def _normalize_chain(cls, chain, side):
        data = chain.copy()
        if missing := [col for col in cls.REQUIRED_COLUMNS if col not in data]:
            raise ValueError(f"{side} table is missing required columns: {missing}")

        data["expiryDate"] = pd.to_datetime(data["expiryDate"]).dt.strftime("%Y-%m-%d")
        data["days_to_expiry"] = pd.to_numeric(data["days_to_expiry"], errors="coerce")
        data["strikePrice"] = pd.to_numeric(data["strikePrice"], errors="coerce")
        data["lastPrice"] = pd.to_numeric(data["lastPrice"], errors="coerce")
        data = data.dropna(subset=["expiryDate", "days_to_expiry", "strikePrice", "lastPrice"])

        if data.empty:
            raise ValueError(f"{side} table has no usable option rows.")
        return data.sort_values("strikePrice").reset_index(drop=True)

    @staticmethod
    def _chain_scalar(chain, column):
        values = chain[column].dropna().unique()
        if len(values) == 0:
            raise ValueError(f"Column {column!r} has no usable values.")
        return values[0]
