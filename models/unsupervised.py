# models/unsupervised.py
# ─────────────────────────────────────────────────────────────────────────────
# Détection de régimes de marché par KMeans
# Fournit un signal basé sur le cluster (momentum moyen du régime)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from config import N_CLUSTERS


class MarketRegimeModel:
    """
    Détecte les régimes de marché via KMeans sur les features agrégées
    (rendement moyen du marché, volatilité moyenne).

    Signal par actif : z-score de momentum relatif au régime détecté.
    """

    def __init__(self, n_clusters: int = N_CLUSTERS):
        self.n_clusters = n_clusters
        self.scaler     = StandardScaler()
        self.kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.is_fitted  = False
        self.cluster_momentum: dict[int, float] = {}

    # ── Entraînement ─────────────────────────────────────────────────────────
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Entraîne sur les rendements quotidiens (dates × actifs).
        Caractéristiques utilisées : rendement moyen & volatilité moyenne.
        """
        mkt_ret  = returns.mean(axis=1).rename("mkt_ret")
        mkt_vol  = returns.std(axis=1).rename("mkt_vol")
        mkt_mom5 = mkt_ret.rolling(5).mean().rename("mkt_mom5")
        X_raw    = pd.concat([mkt_ret, mkt_vol, mkt_mom5], axis=1).dropna()

        X_scaled = self.scaler.fit_transform(X_raw)
        labels   = self.kmeans.fit_predict(X_scaled)

        # Momentum moyen de chaque cluster → sert à ordonner les régimes
        self.cluster_momentum = {
            k: mkt_ret.loc[X_raw.index[labels == k]].mean()
            for k in range(self.n_clusters)
        }
        self._index  = X_raw.index
        self._labels = labels
        self.is_fitted = True

    # ── Prédiction ───────────────────────────────────────────────────────────
    def predict_signal(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Retourne un signal (dates × actifs) dans [-1, 1].
        Basé sur le momentum du régime courant, normalisé.
        """
        mkt_ret  = returns.mean(axis=1)
        mkt_vol  = returns.std(axis=1)
        mkt_mom5 = mkt_ret.rolling(5).mean()
        X_raw    = pd.concat([mkt_ret, mkt_vol, mkt_mom5], axis=1).dropna()
        X_raw.columns = ["mkt_ret", "mkt_vol", "mkt_mom5"]

        X_scaled = self.scaler.transform(X_raw)
        labels   = self.kmeans.predict(X_scaled)

        # Score du régime 
        regime_scores = pd.Series(
            [self.cluster_momentum[l] for l in labels],
            index=X_raw.index,
            name="regime_score",
        )
        # Normalisation [-1, 1]
        rmax = regime_scores.abs().max()
        if rmax > 0:
            regime_scores /= rmax

        signal = pd.DataFrame(
            np.outer(regime_scores.values, np.ones(len(returns.columns))),
            index=regime_scores.index,
            columns=returns.columns,
        )
        return signal
    
    def get_cluster_summary(self) -> dict:
        return self.cluster_momentum


if __name__ == "__main__":
    from data.loader import download_prices
    from features.build_features import compute_returns
    prices, _ = download_prices()
    ret = compute_returns(prices)
    model = MarketRegimeModel()
    model.fit(ret)
    sig = model.predict_signal(ret)
    print(sig.tail())
