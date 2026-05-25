# features/build_features.py
# ─────────────────────────────────────────────────────────────────────────────
# Construction des features : rendements, momentum, volatilité
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from config import MOMENTUM_WINDOWS, VOLATILITY_WINDOW


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendements log-quotidiens."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_momentum(returns: pd.DataFrame, windows: list[int] = MOMENTUM_WINDOWS) -> pd.DataFrame:
    """
    Momentum cumulatif sur plusieurs fenêtres.
    mom_W = somme des rendements log sur W jours.
    """
    frames = []
    for w in windows:
        mom = returns.rolling(w).sum()
        mom.columns = [f"{c}_mom{w}" for c in returns.columns]
        frames.append(mom)
    return pd.concat(frames, axis=1)


def compute_volatility(returns: pd.DataFrame, window: int = VOLATILITY_WINDOW) -> pd.DataFrame:
    """Volatilité réalisée glissante (écart-type des rendements log)."""
    vol = returns.rolling(window).std()
    vol.columns = [f"{c}_vol{window}" for c in returns.columns]
    return vol


def build_feature_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble toutes les features dans un seul DataFrame 'wide'.

    Colonnes : pour chaque actif t :
        t_ret, t_mom5, t_mom20, t_vol20

    Index    : dates (après suppression des NaN initiaux)
    """
    returns    = compute_returns(prices)
    returns.columns = [f"{c}_ret" for c in prices.columns]

    momentum   = compute_momentum(compute_returns(prices))
    volatility = compute_volatility(compute_returns(prices))

    feature_df = pd.concat([returns, momentum, volatility], axis=1).dropna()
    print(f"[Features] Matrice : {feature_df.shape[0]} dates × {feature_df.shape[1]} features")
    return feature_df


def build_asset_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Retourne un dict  ticker → DataFrame(features) en format 'long',
    utile pour les modèles par actif (LSTM, XGBoost).
    """
    ret = compute_returns(prices)
    result = {}
    for ticker in prices.columns:
        r = ret[[ticker]].rename(columns={ticker: "ret"})
        for w in MOMENTUM_WINDOWS:
            r[f"mom{w}"] = r["ret"].rolling(w).sum()
        r[f"vol{VOLATILITY_WINDOW}"] = r["ret"].rolling(VOLATILITY_WINDOW).std()
        r["target"] = r["ret"].shift(-1)   # rendement du lendemain (cible supervisée)
        result[ticker] = r.dropna()
    return result


if __name__ == "__main__":
    from data.loader import download_prices
    prices, _ = download_prices()
    feat = build_feature_matrix(prices)
    print(feat.describe())
