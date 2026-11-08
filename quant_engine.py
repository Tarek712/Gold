import numpy as np
from scipy.stats import norm
from typing import Dict, Any


class XAUQuantEngine:
    """
    النواة الرياضية الكمية لتحليل الذهب (XAU/USD Spot/CFD)
    تشمل: k-NN Historical Matcher, Monte Carlo Ruin Cone, Bayesian Kelly, CVD & Order Flow
    """

    def __init__(self, account_balance: float = 5000.0, fixed_lot: float = 0.01):
        self.account_balance = account_balance
        self.fixed_lot = fixed_lot
        self.max_daily_dd_pct = 0.05
        self.max_total_dd_pct = 0.10

    @staticmethod
    def z_score(series: np.ndarray) -> np.ndarray:
        std = np.std(series)
        if std < 1e-8:
            return np.zeros_like(series)
        return (series - np.mean(series)) / std

    def knn_analog_matcher(
        self, closes: np.ndarray, window: int = 32, horizon: int = 16, k: int = 5
    ) -> Dict[str, Any]:
        n = len(closes)
        if n < window + horizon + 50:
            return {
                "match_score": 0.0,
                "current_norm": [],
                "projected_median": [],
                "projected_upper": [],
                "projected_lower": [],
                "up_count": 0,
                "down_count": 0,
                "expected_move_usd": 0.0,
                "horizon_bars": horizon,
            }

        current_window = closes[-window:]
        current_norm = self.z_score(current_window)
        weights = np.linspace(0.5, 1.5, window)

        distances = []
        search_end = n - window - horizon

        for i in range(search_end):
            hist_win = closes[i : i + window]
            hist_norm = self.z_score(hist_win)
            dist = np.sqrt(np.sum(weights * (current_norm - hist_norm) ** 2))
            distances.append((dist, i))

        distances.sort(key=lambda x: x[0])
        top_k = distances[:k]

        best_dist = top_k[0][0]
        match_score = float(np.clip(np.exp(-best_dist / (window * 0.35)), 0.01, 0.99))

        future_paths = []
        up_count = 0
        down_count = 0

        for _, idx in top_k:
            hist_win = closes[idx : idx + window]
            hist_fut = closes[idx + window : idx + window + horizon]
            combined_std = np.std(hist_win) if np.std(hist_win) > 1e-8 else 1.0
            fut_norm = (hist_fut - hist_win[-1]) / combined_std + current_norm[-1]
            future_paths.append(fut_norm)

            if hist_fut[-1] >= hist_win[-1]:
                up_count += 1
            else:
                down_count += 1

        future_paths_arr = np.array(future_paths)
        median_path = np.median(future_paths_arr, axis=0)
        upper_band = np.percentile(future_paths_arr, 80, axis=0)
        lower_band = np.percentile(future_paths_arr, 20, axis=0)

        curr_std = float(np.std(current_window))
        expected_move_usd = float((median_path[-1] - current_norm[-1]) * curr_std)

        return {
            "match_score": round(match_score, 2),
            "up_count": int(up_count),
            "down_count": int(down_count),
            "expected_move_usd": round(expected_move_usd, 2),
            "horizon_bars": horizon,
            "current_norm": [round(float(x), 3) for x in current_norm],
            "projected_median": [round(float(x), 3) for x in median_path],
            "projected_upper": [round(float(x), 3) for x in upper_band],
            "projected_lower": [round(float(x), 3) for x in lower_band],
        }

    def monte_carlo_ruin_cone(
        self, closes: np.ndarray, num_sims: int = 300, steps: int = 40
    ) -> Dict[str, Any]:
        returns = np.diff(closes) / closes[:-1]
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))
        if sigma < 1e-8:
            sigma = 0.001

        last_price = float(closes[-1])
        dt = 1.0

        shocks = np.random.normal(0, 1, size=(num_sims, steps))
        price_paths = np.zeros((num_sims, steps + 1))
        price_paths[:, 0] = last_price

        for t in range(1, steps + 1):
            price_paths[:, t] = price_paths[:, t - 1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks[:, t - 1]
            )

        ounces = self.fixed_lot * 100.0
        pnl_paths = (price_paths - last_price) * ounces

        ruin_threshold_usd = -self.account_balance * 0.02
        ruin_hits = np.any(pnl_paths <= ruin_threshold_usd, axis=1)
        prob_ruin = float(np.mean(ruin_hits) * 100.0)

        p90 = np.percentile(price_paths, 90, axis=0)
        p75 = np.percentile(price_paths, 75, axis=0)
        p50 = np.percentile(price_paths, 50, axis=0)
        p25 = np.percentile(price_paths, 25, axis=0)
        p10 = np.percentile(price_paths, 10, axis=0)

        return {
            "prob_ruin_pct": round(prob_ruin, 2),
            "p90": [round(float(x), 2) for x in p90],
            "p75": [round(float(x), 2) for x in p75],
            "p50": [round(float(x), 2) for x in p50],
            "p25": [round(float(x), 2) for x in p25],
            "p10": [round(float(x), 2) for x in p10],
        }

    def bayesian_kelly_update(
        self, closes: np.ndarray, volumes: np.ndarray, knn_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        typical_prices = closes[-50:]
        vols = volumes[-50:]
        vwap = float(np.sum(typical_prices * vols) / (np.sum(vols) + 1e-8))
        std_dev = float(np.std(typical_prices)) if np.std(typical_prices) > 1e-8 else 1.0

        z_vwap = (closes[-1] - vwap) / std_dev
        prior_prob = float(np.clip(0.50 + 0.12 * np.tanh(z_vwap), 0.25, 0.75))

        total_k = knn_data["up_count"] + knn_data["down_count"]
        if total_k > 0:
            knn_bull_ratio = (knn_data["up_count"] + 0.5) / (total_k + 1.0)
        else:
            knn_bull_ratio = 0.50

        numerator = knn_bull_ratio * prior_prob
        denominator = numerator + (1.0 - knn_bull_ratio) * (1.0 - prior_prob)
        posterior_prob = float(np.clip(numerator / (denominator + 1e-8), 0.15, 0.88))

        kl_div = float(
            posterior_prob * np.log((posterior_prob + 1e-8) / (prior_prob + 1e-8))
            + (1 - posterior_prob) * np.log((1 - posterior_prob + 1e-8) / (1 - prior_prob + 1e-8))
        )

        b = 2.5
        p = max(posterior_prob, 1.0 - posterior_prob)
        q = 1.0 - p
        kelly_fraction = max(0.0, (p * b - q) / b)
        half_kelly_pct = round(kelly_fraction * 0.5 * 100.0, 2)

        x_grid = np.linspace(0.05, 0.95, 60)
        prior_pdf = norm.pdf(x_grid, loc=prior_prob, scale=0.14)
        post_pdf = norm.pdf(x_grid, loc=posterior_prob, scale=0.08)

        return {
            "vwap": round(vwap, 2),
            "prior_pct": round(prior_prob * 100, 1),
            "posterior_pct": round(posterior_prob * 100, 1),
            "kl_divergence": round(kl_div, 3),
            "half_kelly_pct": half_kelly_pct,
            "recommended_lot": self.fixed_lot,
            "direction": "LONG (BUY)" if posterior_prob >= 0.5 else "SHORT (SELL)",
            "x_grid": [round(float(x), 2) for x in x_grid],
            "prior_curve": [round(float(y), 3) for y in prior_pdf],
            "posterior_curve": [round(float(y), 3) for y in post_pdf],
        }

    def order_flow_cvd_and_smc(
        self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray
    ) -> Dict[str, Any]:
        ranges = np.maximum(highs - lows, 1e-5)
        buy_ratio = (closes - lows) / ranges
        sell_ratio = (highs - closes) / ranges
        delta = volumes * (buy_ratio - sell_ratio)

        cvd = np.cumsum(delta[-60:])
        cvd_norm = cvd - cvd[0]

        bull_ob = float(np.min(lows[-20:]))
        bear_ob = float(np.max(highs[-20:]))

        recent_delta_sum = float(np.sum(delta[-10:]))
        regime = "BULLISH ABSORPTION" if recent_delta_sum > 0 else "BEARISH DISTRIBUTION"

        return {
            "cvd_series": [round(float(x), 1) for x in cvd_norm],
            "delta_bars": [round(float(x), 1) for x in delta[-60:]],
            "current_cvd": round(float(cvd_norm[-1]), 1),
            "regime": regime,
            "bullish_ob": round(bull_ob, 2),
            "bearish_ob": round(bear_ob, 2),
        }
