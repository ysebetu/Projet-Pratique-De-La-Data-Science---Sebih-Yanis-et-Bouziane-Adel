# signals/combine.py
# ─────────────────────────────────────────────────────────────────────────────
# Combinaison des signaux multi-modèles via régression Ridge
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from config import RIDGE_ALPHA


class SignalCombiner:
    """
    Apprend automatiquement comment pondérer les signaux des différents modèles
    pour maximiser leur pouvoir prédictif sur les rendements réalisés.

    Approche :
      - Pour chaque date t et chaque actif i, on dispose de K signaux.
      - X : matrice (n_obs × K) des signaux empilés (toutes dates, tous actifs)
      - y : vecteur des rendements réalisés le lendemain
      - Régression Ridge → coefficients λ_k
      - Signal combiné : Σ_k λ_k · signal_k
    """

    def __init__(self, alpha: float = RIDGE_ALPHA):
        self.alpha   = alpha
        self.ridge   = Ridge(alpha=alpha, fit_intercept=True)
        self.scaler  = StandardScaler()
        self.weights_: np.ndarray | None = None
        self.signal_names: list[str] = []

    # ── Construction de la matrice X ─────────────────────────────────────────
    @staticmethod
    def _stack_signals(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        signals : dict name → DataFrame(dates × tickers)
        Retourne un DataFrame long avec MultiIndex (date, ticker) strict.
        """
        frames = []
        for name, df in signals.items():
            # stack() crée un MultiIndex (date, ticker_colname)
            melted = df.stack(future_stack=True).rename(name)
            # Normaliser les noms des niveaux : niveau 0 = "date", niveau 1 = "ticker"
            melted.index.names = ["date", "ticker"]
            frames.append(melted)
        combined = pd.concat(frames, axis=1).dropna()
        combined.index.names = ["date", "ticker"]
        return combined

    # ── Entraînement ─────────────────────────────────────────────────────────
    def fit(
        self,
        signals: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        train_end: pd.Timestamp | str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        signals   : dict model_name → signal DataFrame (dates × tickers)
        returns   : rendements journaliers (dates × tickers)
        train_end : date limite de la période d'entraînement (exclu si None → 70%)
        """
        combined = self._stack_signals(signals)
        self.signal_names = list(signals.keys())

        # Rendements du lendemain en format long — index forcé à (date, ticker)
        ret_shifted = returns.shift(-1).copy()
        ret_shifted.index.name = "date"
        ret_long = ret_shifted.stack(future_stack=True).rename("target")
        ret_long.index.names = ["date", "ticker"]

        # Jointure sur le MultiIndex commun (date, ticker)
        # On passe par reset_index pour éviter tout problème de noms de niveaux
        df_combined = combined.reset_index()   # colonnes: date, ticker, signal...
        df_ret      = ret_long.reset_index()   # colonnes: date, ticker, target
        data = df_combined.merge(df_ret, on=["date", "ticker"], how="inner").dropna()

        if train_end is None:
            split_idx  = int(len(data) * 0.7)
            train_data = data.iloc[:split_idx]
        else:
            train_data = data.loc[data["date"] <= train_end]

        X_raw = train_data[self.signal_names].values
        y     = train_data["target"].values

        X_scaled = self.scaler.fit_transform(X_raw)
        self.ridge.fit(X_scaled, y)
        self.weights_ = self.ridge.coef_

        print(f"[Ridge] Poids appris : { dict(zip(self.signal_names, self.weights_.round(4))) }")

    # ── Prédiction ───────────────────────────────────────────────────────────
    def predict(self, signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine les signaux avec les poids Ridge appris.
        Retourne un DataFrame (dates × tickers) normalisé en rang.
        """
        if self.weights_ is None:
            raise RuntimeError("Appeler .fit() avant .predict()")

        combined = self._stack_signals(signals)
        X_raw    = combined[self.signal_names].values
        X_scaled = self.scaler.transform(X_raw)
        scores   = self.ridge.predict(X_scaled)

        # Reconstruction du DataFrame dates × tickers
        index = combined.index   # MultiIndex (date, ticker)
        index.names = ["date", "ticker"]
        score_s  = pd.Series(scores, index=index, name="score")
        score_df = score_s.unstack("ticker")   # dates × tickers

        # Normalisation en rang cross-sectionnel → [-1, 1]
        ranked = score_df.rank(axis=1, pct=True) * 2 - 1
        return ranked

    # ── Résumé des poids ─────────────────────────────────────────────────────
    def summary(self) -> pd.Series:
        if self.weights_ is None:
            raise RuntimeError("Modèle non entraîné.")
        return pd.Series(self.weights_, index=self.signal_names, name="ridge_weight")


if __name__ == "__main__":
    # Démo rapide avec des signaux aléatoires
    import numpy as np
    dates   = pd.date_range("2020-01-01", periods=500, freq="B")
    tickers = ["A", "B", "C", "D", "E"]
    rng     = np.random.default_rng(42)

    sig1 = pd.DataFrame(rng.standard_normal((500, 5)), index=dates, columns=tickers)
    sig2 = pd.DataFrame(rng.standard_normal((500, 5)), index=dates, columns=tickers)
    rets = pd.DataFrame(rng.standard_normal((500, 5)) * 0.01, index=dates, columns=tickers)

    combiner = SignalCombiner()
    combiner.fit({"model1": sig1, "model2": sig2}, rets)
    final_signal = combiner.predict({"model1": sig1, "model2": sig2})
    print(final_signal.tail())
    print(combiner.summary())
