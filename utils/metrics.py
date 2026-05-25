# utils/metrics.py
# ─────────────────────────────────────────────────────────────────────────────
# Métriques de performance des portefeuilles
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


TRADING_DAYS = 252   # jours de bourse par an


def annualized_return(returns: pd.Series) -> float:
    """Rendement annualisé"""
    total = (1 + returns).prod()
    n     = len(returns)
    return float(total ** (TRADING_DAYS / n) - 1) if n > 0 else np.nan


def annualized_volatility(returns: pd.Series) -> float:
    """Volatilité annualisée (écart-type × √252)."""
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.02) -> float:
    """Ratio de Sharpe annualisé (taux sans risque = 2% par défaut)."""
    excess = annualized_return(returns) - risk_free
    vol    = annualized_volatility(returns)
    return float(excess / vol) if vol > 0 else np.nan


def sortino_ratio(returns: pd.Series, risk_free: float = 0.02) -> float:
    """Ratio de Sortino (uniquement la volatilité à la baisse)."""
    annual_ret = annualized_return(returns)
    excess     = annual_ret - risk_free
    downside   = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    return float(excess / downside) if downside > 0 else np.nan


def max_drawdown(returns: pd.Series) -> float:
    """Maximum Drawdown (valeur négative)."""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    """Ratio de Calmar = rendement annualisé / |max drawdown|."""
    mdd = max_drawdown(returns)
    ar  = annualized_return(returns)
    return float(ar / abs(mdd)) if mdd != 0 else np.nan


def win_rate(returns: pd.Series) -> float:
    """Proportion de jours avec un rendement positif."""
    return float((returns > 0).mean())


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Information Ratio = rendement actif moyen / tracking error."""
    active = strategy_returns - benchmark_returns.reindex(strategy_returns.index).fillna(0)
    ar     = active.mean() * TRADING_DAYS
    te     = active.std()  * np.sqrt(TRADING_DAYS)
    return float(ar / te) if te > 0 else np.nan


def compute_all_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free: float = 0.02,
    name: str = "Strategy",
) -> pd.Series:
    """
    Calcule toutes les métriques de performance et retourne une Series.
    """
    metrics = {
        "Rendement annualisé (%)": annualized_return(returns) * 100,
        "Volatilité annualisée (%)": annualized_volatility(returns) * 100,
        "Sharpe Ratio":  sharpe_ratio(returns, risk_free),
        "Sortino Ratio": sortino_ratio(returns, risk_free),
        "Max Drawdown (%)": max_drawdown(returns) * 100,
        "Calmar Ratio": calmar_ratio(returns),
        "Win Rate (%)": win_rate(returns) * 100,
    }
    if benchmark_returns is not None:
        metrics["Information Ratio"] = information_ratio(returns, benchmark_returns)

    return pd.Series(metrics, name=name).round(3)


def performance_table(
    strategies: dict[str, pd.Series],
    benchmark_returns: pd.Series | None = None,
    risk_free: float = 0.02,
) -> pd.DataFrame:
    """
    strategies : dict name → rendements journaliers
    Retourne un tableau comparatif.
    """
    rows = [
        compute_all_metrics(ret, benchmark_returns, risk_free, name=name)
        for name, ret in strategies.items()
    ]
    return pd.DataFrame(rows).T


def rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    """Sharpe glissant sur `window` jours."""
    roll_ret = returns.rolling(window).mean() * TRADING_DAYS
    roll_vol = returns.rolling(window).std()  * np.sqrt(TRADING_DAYS)
    return (roll_ret / roll_vol).replace([np.inf, -np.inf], np.nan)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Série de drawdown (entre 0 et -1)."""
    cumulative  = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    return (cumulative - rolling_max) / rolling_max


if __name__ == "__main__":
    rng  = np.random.default_rng(42)
    ret  = pd.Series(rng.normal(0.0005, 0.012, 1000))
    bret = pd.Series(rng.normal(0.0003, 0.010, 1000))

    print(compute_all_metrics(ret, bret, name="Ma Stratégie").to_string())
