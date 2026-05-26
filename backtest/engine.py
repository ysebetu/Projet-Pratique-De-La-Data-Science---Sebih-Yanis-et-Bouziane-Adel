# backtest/engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Moteur de backtest : simulation quotidienne avec coûts de transaction
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from config import TRANSACTION_COST, INITIAL_CAPITAL


class BacktestEngine:
    """
    Simule un portefeuille pondéré sur des données historiques.

    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        transaction_cost: float = TRANSACTION_COST,
    ):
        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost

    # ── Simulation ────────────────────────────────────────────────────────────
    def run(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        name: str = "Strategy",
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        weights : DataFrame (dates × tickers) — poids à t (décidés avant t+1)
        returns : DataFrame (dates × tickers) — rendements log à t
        name    : étiquette de la stratégie

        Retourne un DataFrame avec colonnes :
          portfolio_return, gross_return, transaction_costs, nav
        """
        # Alignement
        common_dates   = weights.index.intersection(returns.index)
        common_tickers = weights.columns.intersection(returns.columns)

        w = weights.loc[common_dates, common_tickers].astype(float)
        r = returns.loc[common_dates, common_tickers].astype(float)

        # Rendements bruts du portefeuille (avant coûts)
        gross_returns = (w.shift(1) * r).sum(axis=1)   # shift : poids décidés à t-1

        # Coûts de transaction (variation des poids)
        turnover    = w.diff().abs().sum(axis=1)
        tc_cost     = turnover * self.transaction_cost
        net_returns = gross_returns - tc_cost

        # NAV
        nav = (1 + net_returns).cumprod() * self.initial_capital

        result = pd.DataFrame({
            "portfolio_return"  : net_returns,
            "gross_return"      : gross_returns,
            "transaction_costs" : tc_cost,
            "nav"               : nav,
        }, index=common_dates)

        print(
            f"[Backtest] '{name}' terminé — "
            f"{len(result)} jours | NAV finale = {nav.iloc[-1]:,.0f} €"
        )
        return result

    # ── Buy-and-Hold benchmark ────────────────────────────────────────────────
    def run_buyhold(
        self,
        benchmark_returns: pd.Series,
        name: str = "Buy & Hold CAC 40",
    ) -> pd.DataFrame:
        nav = (1 + benchmark_returns).cumprod() * self.initial_capital
        result = pd.DataFrame({
            "portfolio_return"  : benchmark_returns,
            "gross_return"      : benchmark_returns,
            "transaction_costs" : 0.0,
            "nav"               : nav,
        })
        print(f"[Backtest] '{name}' calculé.")
        return result

    # ── Comparaison multi-stratégies ──────────────────────────────────────────
    @staticmethod
    def compare_nav(*results: tuple[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Retourne un DataFrame de NAV alignées (dates × stratégies).
        """
        frames = {name: df["nav"] for name, df in results}
        return pd.DataFrame(frames).dropna()


if __name__ == "__main__":
    from data.loader import download_prices
    from features.build_features import compute_returns
    from portfolio.weights import PortfolioConstructor
    import numpy as np

    prices, bench = download_prices()
    ret = compute_returns(prices)

    rng    = np.random.default_rng(0)
    signal = pd.DataFrame(
        rng.standard_normal(prices.shape),
        index=prices.index, columns=prices.columns,
    )

    pc      = PortfolioConstructor(top_n=5, rebalance_freq="W")
    weights = pc.compute_weights(signal)

    engine  = BacktestEngine()
    strat   = engine.run(weights, ret, name="Test Signal")
    bh      = engine.run_buyhold(bench.pct_change().dropna(), name="CAC 40 B&H")

    nav_compare = BacktestEngine.compare_nav(
        ("Strategy", strat),
        ("CAC 40 B&H", bh),
    )
    print(nav_compare.tail())
