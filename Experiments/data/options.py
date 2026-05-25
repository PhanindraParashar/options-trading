import pandas as pd
import numpy as np
from jugaad_data.nse import NSELive
from nselib import derivatives
from py_vollib_vectorized import (
    vectorized_delta, vectorized_gamma, vectorized_theta,
    vectorized_vega, vectorized_rho, vectorized_implied_volatility,
)
import warnings
warnings.filterwarnings("ignore")

OPTION_COLS = {
    'impliedVolatility': 'IV',
    'lastPrice': 'LTP',
    'openInterest': 'OI',
    'totalTradedVolume': 'Volume',
}
GREEK_COLS = ['delta', 'gamma', 'theta', 'vega', 'rho']
FINAL_COLS = [
    'expiryDate', 'days_to_expiry', 'strikePrice', 'impliedVolatility',
    *GREEK_COLS,
    'lastPrice', 'openInterest', 'totalTradedVolume',
]
CHAIN_COLS = {
    'CE': {name: f'CALLS_{source}' for name, source in OPTION_COLS.items()},
    'PE': {name: f'PUTS_{source}' for name, source in OPTION_COLS.items()},
}
NUMERIC_COLS = ['Strike_Price', *CHAIN_COLS['CE'].values(), *CHAIN_COLS['PE'].values()]


class OptionsChain:
    def __init__(self, derivative="NIFTY", LOT_SIZE=65):
        self.derivative = derivative
        self.LOT_SIZE = LOT_SIZE

    # ---------- option tables ----------

    def get_ce(self, expiry=None, r=0.07, recompute_iv=False, min_traded_qty=100):
        return self._build('CE', 'c', expiry, r, recompute_iv, min_traded_qty)

    def get_pe(self, expiry=None, r=0.07, recompute_iv=False, min_traded_qty=100):
        return self._build('PE', 'p', expiry, r, recompute_iv, min_traded_qty)

    # ---------- quick lookups ----------

    def spot(self):
        rows, spot, _ = self._select()
        return spot

    def expiries(self):
        """All available expiry dates, sorted ascending, as yyyy-mm-dd strings."""
        raw = NSELive().index_option_chain(self.derivative)['records']['expiryDates']
        return [pd.to_datetime(d).strftime('%Y-%m-%d') for d in raw]

    def nearest_expiry(self):
        return pd.to_datetime(self.expiries()[0]).strftime('%Y-%m-%d')

    def days_to_nearest_expiry(self):
        expiry_dt = pd.to_datetime(self.nearest_expiry()).replace(hour=15, minute=30)
        return (expiry_dt - pd.Timestamp.now()).total_seconds() / 86400

    def strikes(self, expiry=None):
        rows, _, _ = self._select(expiry)
        return sorted(rows['Strike_Price'].dropna().unique())

    def atm_strike(self, expiry=None):
        rows, spot, _ = self._select(expiry)
        strikes = rows['Strike_Price'].dropna().unique()
        return min(strikes, key=lambda k: abs(k - spot))

    # ---------- internals ----------

    def _raw(self, expiry=None):
        kwargs = {'symbol': self.derivative}
        if expiry is not None:
            kwargs['expiry_date'] = pd.to_datetime(expiry).strftime('%d-%m-%Y')

        df = derivatives.nse_live_option_chain(**kwargs)
        if df.empty:
            raise ValueError(f"No option-chain rows returned for {self.derivative}.")

        df = df.copy()
        df['expiryDate'] = pd.to_datetime(
            df['Expiry_Date'], dayfirst=True, errors='coerce'
        ).dt.normalize()

        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = self._to_numeric(df[col])

        return df

    def _select(self, expiry=None):
        """Return (rows, spot, expiry_dt) for the given expiry, or nearest if None."""
        if expiry is None:
            chosen_dt = pd.to_datetime(self.nearest_expiry()).normalize()
        else:
            chosen_dt = pd.to_datetime(expiry).normalize()

        raw = self._raw(chosen_dt)
        target_date = chosen_dt.normalize()
        rows = raw[raw['expiryDate'] == target_date].copy()

        if rows.empty:
            raise ValueError(
                f"No rows for expiry {target_date.strftime('%Y-%m-%d')} "
                f"in NSE response."
            )

        expiry_dt = chosen_dt.replace(hour=15, minute=30)
        spot = self._spot_from_chain(rows)
        return rows, spot, expiry_dt

    @staticmethod
    def _to_numeric(series):
        return pd.to_numeric(
            series.astype(str).str.replace(',', '', regex=False).replace('-', np.nan),
            errors='coerce',
        )

    def _spot_from_chain(self, df):
        spot_cols = [
            'Value', 'Underlying_Value', 'Underlying Value',
            'underlyingValue', 'underlying', 'Spot_Price',
        ]
        for col in spot_cols:
            if col in df.columns:
                spot = self._to_numeric(df[col]).dropna()
                if not spot.empty:
                    return float(spot.iloc[0])

        parity = df[['Strike_Price', 'CALLS_LTP', 'PUTS_LTP']].dropna()
        parity = parity[(parity['CALLS_LTP'] > 0) & (parity['PUTS_LTP'] > 0)]
        if parity.empty:
            raise ValueError(
                "Could not determine spot price from nselib option-chain data. "
                "Expected a spot/value column or usable CALLS_LTP/PUTS_LTP pairs."
            )

        atm = parity.loc[(parity['CALLS_LTP'] - parity['PUTS_LTP']).abs().idxmin()]
        return float(atm['Strike_Price'] + atm['CALLS_LTP'] - atm['PUTS_LTP'])

    def _build(self, key, flag, expiry, r, recompute_iv, min_traded_qty):
        rows, spot, expiry_dt = self._select(expiry)
        days = (expiry_dt - pd.Timestamp.now()).total_seconds() / 86400
        T = max(days / 365, 1 / (365 * 24))

        mapping = CHAIN_COLS[key]
        required = ['Strike_Price', *mapping.values()]
        if missing := [c for c in required if c not in rows.columns]:
            raise ValueError(f"Missing expected nselib columns for {key}: {missing}")

        df = rows[required].rename(
            columns={'Strike_Price': 'strikePrice', **{v: k for k, v in mapping.items()}}
        )
        if df.empty:
            raise ValueError(
                f"No {key} rows for expiry {expiry_dt.strftime('%Y-%m-%d')} "
                f"on {self.derivative}."
            )
        df = df.dropna(subset=[
            'strikePrice', 'impliedVolatility', 'lastPrice',
            'openInterest', 'totalTradedVolume',
        ])
        df = df[df['totalTradedVolume'] > min_traded_qty].reset_index(drop=True)
        df['expiryDate'] = expiry_dt.normalize()
        df['days_to_expiry'] = days

        if df.empty:
            raise ValueError(
                f"No {key} rows for expiry {expiry_dt.strftime('%Y-%m-%d')} "
                f"after applying min_traded_qty={min_traded_qty}."
            )

        n = len(df)
        S = np.full(n, spot, dtype=float)
        K = df['strikePrice'].astype(float).values
        t = np.full(n, T, dtype=float)
        rr = np.full(n, r, dtype=float)
        flags = np.full(n, flag)

        if recompute_iv:
            price = df['lastPrice'].astype(float).values
            sigma = vectorized_implied_volatility(price, S, K, t, rr, flags, return_as='numpy')
            df['impliedVolatility'] = sigma * 100
        else:
            sigma = df['impliedVolatility'].astype(float).values / 100.0

        kw = dict(return_as='numpy')
        df['delta'] = vectorized_delta(flags, S, K, t, rr, sigma, **kw)
        df['gamma'] = vectorized_gamma(flags, S, K, t, rr, sigma, **kw)
        df['theta'] = vectorized_theta(flags, S, K, t, rr, sigma, **kw)
        df['vega']  = vectorized_vega (flags, S, K, t, rr, sigma, **kw)
        df['rho']   = vectorized_rho  (flags, S, K, t, rr, sigma, **kw)

        return df[FINAL_COLS].round(4).sort_values('strikePrice').reset_index(drop=True)
