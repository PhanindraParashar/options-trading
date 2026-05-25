import math

import numpy as np
import pandas as pd

try:
    from scipy.interpolate import UnivariateSpline
    from scipy.special import ndtr
except ImportError:  # pragma: no cover - notebook environments may vary
    UnivariateSpline = None
    ndtr = None


class BreedenLitzenberger:
    """
    Estimate risk-neutral expiry distributions implied by option prices.

    This is an implied distribution extracted from option-chain prices. It is
    not a real-world forecast distribution.
    """

    def __init__(
        self,
        risk_free_rate=0.07,
        n_grid=500,
        n_samples=10_000,
        random_state=42,
        min_paired_strikes=8,
        quantile_probabilities=None,
    ):
        self.risk_free_rate = risk_free_rate
        self.n_grid = n_grid
        self.n_samples = n_samples
        self.random_state = random_state
        self.min_paired_strikes = min_paired_strikes
        self.quantile_probabilities = quantile_probabilities or [0.05, 0.25, 0.50, 0.75, 0.95]

    def estimate(
        self,
        CE,
        PE,
        expiry=None,
        risk_free_rate=None,
        n_grid=None,
        n_samples=None,
        random_state=None,
    ):
        """
        Estimate one implied distribution from one CE chain and one PE chain.
        """
        ce = self._normalize_chain(CE, side="CE")
        pe = self._normalize_chain(PE, side="PE")

        merged = pd.merge(
            ce,
            pe,
            on=["expiryDate", "strikePrice"],
            suffixes=("_call", "_put"),
            how="inner",
        ).sort_values("strikePrice")
        if len(merged) < self.min_paired_strikes:
            raise ValueError("Not enough paired strikes in CE and PE.")

        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        r = self.risk_free_rate if risk_free_rate is None else risk_free_rate
        grid_size = self.n_grid if n_grid is None else n_grid
        sample_size = self.n_samples if n_samples is None else n_samples
        expiry_key = self._expiry_label(merged, expiry)

        data = self._prepare_merged_chain(merged)
        rows_before = len(data)
        data = data[data["is_valid"]].copy()
        if len(data) < self.min_paired_strikes:
            raise ValueError("Not enough clean paired strikes.")

        T = float(data["days_to_expiry"].iloc[0]) / 365.0
        D = float(np.exp(-r * T))

        data["forward_i"] = data["strikePrice"] + (data["call_price"] - data["put_price"]) / D
        data["liquidity_weight"] = self._liquidity_weight(data)

        forward_initial = self._weighted_median(
            data["forward_i"].to_numpy(),
            data["liquidity_weight"].to_numpy(),
        )
        keep = self._drop_parity_outliers(data["forward_i"], data["liquidity_weight"], forward_initial)
        data = data[keep].copy()
        if len(data) < self.min_paired_strikes:
            raise ValueError("Too few strikes after parity outlier removal.")

        F = self._weighted_median(data["forward_i"].to_numpy(), data["liquidity_weight"].to_numpy())
        data = self._add_synthetic_curve_inputs(data, F, D, T)

        x = data["log_moneyness"].to_numpy()
        w = data["total_variance"].to_numpy()
        weights = data["curve_weight"].to_numpy()
        w_smooth = self._smooth_total_variance(x, w, weights)
        sigma_smooth = np.sqrt(np.clip(w_smooth / T, 1e-8, None))

        strikes = data["strikePrice"].to_numpy(dtype=float)
        grid = np.linspace(strikes.min(), strikes.max(), grid_size)
        grid_x = np.log(grid / F)
        grid_sigma = np.interp(grid_x, x, sigma_smooth)
        smooth_calls = self._black76_call(F=F, K=grid, T=T, r=r, sigma=grid_sigma)

        density_raw = np.gradient(np.gradient(smooth_calls, grid), grid) / D
        density, warning, neg_share, large_neg_share = self._clean_density(density_raw)
        density = self._normalize_density(density, grid)

        cdf = self._cdf_from_density(density, grid)
        interval_probability = self._interval_probability(cdf)
        mean = float(np.trapz(grid * density, grid))
        variance = float(np.trapz((grid - mean) ** 2 * density, grid))
        std = float(np.sqrt(variance))
        quantiles = self._quantiles_from_cdf(grid, cdf, self.quantile_probabilities)
        samples = self._sample_from_cdf(grid, cdf, sample_size, rng)

        return {
            "expiry": expiry_key,
            "note": "Approximate risk-neutral distribution implied by option prices; not a real-world physical forecast.",
            "strike_grid": grid,
            "probability_density": density,
            "cumulative_probability": cdf,
            "interval_probability": pd.DataFrame(
                {
                    "strike_left": grid[:-1],
                    "strike_right": grid[1:],
                    "probability": interval_probability,
                }
            ),
            "mean_implied_expiry_level": mean,
            "standard_deviation": std,
            "quantiles": quantiles,
            "samples": samples,
            "diagnostics": {
                "expiry": expiry_key,
                "forward": float(F),
                "discount_factor": D,
                "time_to_expiry": T,
                "rows_used": len(data),
                "rows_dropped": rows_before - len(data),
                "negative_density_share": neg_share,
                "large_negative_density_share": large_neg_share,
                "warning": warning,
            },
            "clean_chain": data,
            "smooth_call_curve": pd.DataFrame(
                {
                    "strikePrice": grid,
                    "callPrice": smooth_calls,
                    "density": density,
                    "cdf": cdf,
                }
            ),
        }

    def implied_vol_from_distribution(self, dist, option_chain):
        """
        Annualized volatility implied by sampled expiry levels.

        `option_chain` can be either the CE or PE table returned by OptionsChain.
        """
        samples = dist["samples"]
        current_price = self._chain_scalar(option_chain, "currentPrice")
        days_to_expiry = self._chain_scalar(option_chain, "days_to_expiry")
        return self.implied_vol_from_simulated_prices(samples, current_price, days_to_expiry)

    @staticmethod
    def implied_vol_from_simulated_prices(sim_prices, S0, days_to_expiry):
        sim_prices = np.asarray(sim_prices, dtype=float)
        sim_prices = sim_prices[sim_prices > 0]

        T = float(days_to_expiry) / 365.0
        if T <= 0:
            raise ValueError("days_to_expiry must be positive.")

        log_returns = np.log(sim_prices / float(S0))
        sigma_T = np.std(log_returns, ddof=1)
        return float(sigma_T / np.sqrt(T))

    @staticmethod
    def _normalize_chain(df, side):
        data = df.copy()
        if missing := [col for col in ["expiryDate", "strikePrice"] if col not in data]:
            raise ValueError(f"{side} is missing columns needed to align chains: {missing}")

        data["expiryDate"] = pd.to_datetime(data["expiryDate"]).dt.strftime("%Y-%m-%d")
        data["strikePrice"] = pd.to_numeric(data["strikePrice"], errors="coerce")
        return data

    @staticmethod
    def _expiry_label(data, expiry):
        if expiry is not None:
            return pd.to_datetime(expiry).strftime("%Y-%m-%d")
        return data["expiryDate"].iloc[0]

    def _prepare_merged_chain(self, df):
        data = df.copy()
        if "days_to_expiry" not in data:
            data["days_to_expiry"] = self._coalesce_columns(
                data,
                ["days_to_expiry_call", "days_to_expiry_put"],
            )

        data = data.rename(columns={"lastPrice_call": "call_price", "lastPrice_put": "put_price"})
        numeric_cols = [
            "days_to_expiry",
            "strikePrice",
            "impliedVolatility_call",
            "impliedVolatility_put",
            "call_price",
            "put_price",
            "openInterest_call",
            "openInterest_put",
            "totalTradedVolume_call",
            "totalTradedVolume_put",
        ]
        for col in numeric_cols:
            if col not in data:
                raise ValueError(f"Merged option chain is missing required column: {col}")
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data["iv_call"] = self._iv_to_decimal(data["impliedVolatility_call"])
        data["iv_put"] = self._iv_to_decimal(data["impliedVolatility_put"])
        data["is_valid"] = (
            np.isfinite(data[numeric_cols]).all(axis=1)
            & (data["call_price"] > 0)
            & (data["put_price"] > 0)
            & data["iv_call"].between(0.01, 5.0)
            & data["iv_put"].between(0.01, 5.0)
            & (data["strikePrice"] > 0)
            & (data["days_to_expiry"] > 0)
        )
        return data

    @staticmethod
    def _coalesce_columns(data, columns):
        if existing := [col for col in columns if col in data]:
            return data[existing].bfill(axis=1).iloc[:, 0]
        raise ValueError(f"None of these columns were found: {columns}")

    @staticmethod
    def _chain_scalar(option_chain, column):
        if column not in option_chain:
            raise ValueError(f"Option chain is missing required column: {column}")

        values = pd.to_numeric(option_chain[column], errors="coerce").dropna().unique()
        if len(values) == 0:
            raise ValueError(f"Option chain column has no valid values: {column}")
        return float(values[0])

    @staticmethod
    def _iv_to_decimal(iv):
        values = pd.to_numeric(iv, errors="coerce").astype(float)
        return np.where(values > 3.0, values / 100.0, values)

    @staticmethod
    def _liquidity_weight(data):
        volume = data["totalTradedVolume_call"].clip(lower=0) + data["totalTradedVolume_put"].clip(lower=0)
        oi = data["openInterest_call"].clip(lower=0) + data["openInterest_put"].clip(lower=0)
        return np.sqrt(volume.to_numpy(dtype=float) + 1.0) * np.sqrt(oi.to_numpy(dtype=float) + 1.0)

    @staticmethod
    def _weighted_median(values, weights):
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not valid.any():
            return float(np.nanmedian(values))

        values = values[valid]
        weights = weights[valid]
        order = np.argsort(values)
        values = values[order]
        weights = weights[order]
        cutoff = 0.5 * weights.sum()
        return float(values[np.searchsorted(np.cumsum(weights), cutoff)])

    def _drop_parity_outliers(self, forward_i, weights, forward):
        deviations = np.abs(forward_i - forward)
        mad = self._weighted_median(deviations.to_numpy(), weights.to_numpy())
        robust_scale = max(1.4826 * mad, 0.0025 * abs(forward), 1.0)
        return deviations <= 6.0 * robust_scale

    @staticmethod
    def _add_synthetic_curve_inputs(data, F, D, T):
        out = data.copy()
        call_converted_from_put = out["put_price"] + D * (F - out["strikePrice"])

        call_liq = np.sqrt(out["totalTradedVolume_call"].clip(lower=0) + 1.0)
        put_liq = np.sqrt(out["totalTradedVolume_put"].clip(lower=0) + 1.0)
        atm_band = max(np.median(np.diff(np.sort(out["strikePrice"].unique()))) * 2.0, F * 0.005)
        atm_weight = np.exp(-0.5 * ((out["strikePrice"] - F) / atm_band) ** 2)
        liq_call_share = call_liq / (call_liq + put_liq)

        # Away from ATM, use the OTM side. Near ATM, blend by liquidity.
        blend_call_weight = np.where(out["strikePrice"] >= F, 1.0, 0.0)
        blend_call_weight = (1.0 - atm_weight) * blend_call_weight + atm_weight * liq_call_share

        out["synthetic_call_price"] = (
            blend_call_weight * out["call_price"]
            + (1.0 - blend_call_weight) * call_converted_from_put
        )
        out["synthetic_iv"] = blend_call_weight * out["iv_call"] + (1.0 - blend_call_weight) * out["iv_put"]
        out["log_moneyness"] = np.log(out["strikePrice"] / F)
        out["total_variance"] = out["synthetic_iv"] ** 2 * T
        out["curve_weight"] = blend_call_weight * call_liq + (1.0 - blend_call_weight) * put_liq
        out = out[out["synthetic_call_price"] > 0].sort_values("log_moneyness")
        return out

    @staticmethod
    def _smooth_total_variance(x, total_variance, weights):
        x = np.asarray(x, dtype=float)
        total_variance = np.asarray(total_variance, dtype=float)
        weights = np.asarray(weights, dtype=float)
        valid = np.isfinite(x) & np.isfinite(total_variance) & np.isfinite(weights) & (total_variance > 0)
        x, total_variance, weights = x[valid], total_variance[valid], weights[valid]

        order = np.argsort(x)
        x, total_variance, weights = x[order], total_variance[order], weights[order]
        _, unique_idx = np.unique(x, return_index=True)
        x, total_variance, weights = x[unique_idx], total_variance[unique_idx], weights[unique_idx]

        if UnivariateSpline is None or len(x) < 10:
            return pd.Series(total_variance).rolling(5, center=True, min_periods=1).median().to_numpy()

        normalized_weights = weights / np.nanmedian(weights)
        smoothing = len(x) * np.nanvar(total_variance) * 0.25
        spline = UnivariateSpline(x, total_variance, w=np.sqrt(normalized_weights), s=smoothing, k=3)
        smoothed = spline(x)
        return np.clip(smoothed, np.nanmin(total_variance) * 0.25, np.nanmax(total_variance) * 4.0)

    def _black76_call(self, F, K, T, r, sigma):
        F = float(F)
        K = np.asarray(K, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        D = np.exp(-r * T)
        vol_sqrt_t = np.maximum(sigma * np.sqrt(T), 1e-8)
        d1 = (np.log(F / K) + 0.5 * vol_sqrt_t**2) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        return D * (F * self._norm_cdf(d1) - K * self._norm_cdf(d2))

    @staticmethod
    def _norm_cdf(x):
        if ndtr is not None:
            return ndtr(x)
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))

    @staticmethod
    def _clean_density(density):
        density = np.asarray(density, dtype=float)
        finite_density = density[np.isfinite(density)]
        if finite_density.size == 0:
            raise ValueError("Density calculation produced no finite values.")

        positive_scale = np.nanmedian(finite_density[finite_density > 0]) if np.any(finite_density > 0) else 1.0
        small_negative = (density < 0) & (density > -0.05 * positive_scale)
        large_negative = density < -0.05 * positive_scale
        neg_share = float(np.mean(density < 0))
        large_neg_share = float(np.mean(large_negative))

        warning = None
        if large_neg_share > 0.05:
            warning = (
                "Many large negative densities remain; option chain may be too noisy "
                "or smoothing is insufficient."
            )

        cleaned = density.copy()
        cleaned[small_negative] = 0.0
        cleaned[large_negative] = 0.0
        cleaned[~np.isfinite(cleaned)] = 0.0
        return cleaned, warning, neg_share, large_neg_share

    @staticmethod
    def _normalize_density(density, grid):
        area = float(np.trapz(density, grid))
        if not np.isfinite(area) or area <= 0:
            raise ValueError("Density has non-positive integral; cannot normalize.")
        return density / area

    @staticmethod
    def _cdf_from_density(density, grid):
        increments = 0.5 * (density[1:] + density[:-1]) * np.diff(grid)
        cdf = np.r_[0.0, np.cumsum(increments)]
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]
        return np.clip(cdf, 0.0, 1.0)

    @staticmethod
    def _interval_probability(cdf):
        return np.diff(cdf)

    @staticmethod
    def _quantiles_from_cdf(grid, cdf, probs):
        return {f"{int(p * 100)}%": float(np.interp(p, cdf, grid)) for p in probs}

    @staticmethod
    def _sample_from_cdf(grid, cdf, n_samples, rng):
        u = rng.uniform(size=n_samples)
        return np.interp(u, cdf, grid)
