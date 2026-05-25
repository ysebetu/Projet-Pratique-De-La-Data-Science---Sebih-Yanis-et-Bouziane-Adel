# portfolio/weights.py
# ─────────────────────────────────────────────────────────────────────────────
# Construction des poids du portefeuille par pondération Softmax
#
#   On attribue un poids à TOUTES les actions proportionnel
#   à leur score via une fonction softmax.
#
#   La softmax transforme n'importe quel vecteur de scores en une distribution
#   de probabilités (poids positifs, somme = 1) :
#
#       w_i = exp(T × s_i) / Σ_j exp(T × s_j)
#
#   où T est la température :
#     - T élevé  → concentration forte sur les meilleures actions
#     - T faible → poids plus uniformes (proche de l'équipondéré)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from scipy.special import softmax as scipy_softmax
from config import SOFTMAX_TEMPERATURE, MAX_WEIGHT_PER_ASSET, REBALANCE_FREQ


class PortfolioConstructor:

    def __init__(
        self,
        temperature    : float = SOFTMAX_TEMPERATURE,
        max_weight     : float = MAX_WEIGHT_PER_ASSET,
        rebalance_freq : str   = REBALANCE_FREQ,
        long_short     : bool  = False,
    ):
        self.temperature    = temperature
        self.max_weight     = max_weight
        self.rebalance_freq = rebalance_freq
        self.long_short     = long_short

    def _softmax_weights(self, signal_row: pd.Series) -> pd.Series:
        valid = signal_row.dropna()
        if len(valid) == 0:
            return pd.Series(0.0, index=signal_row.index)

        if not self.long_short:
            scores = valid.values * self.temperature
            w = scipy_softmax(scores)
            w = pd.Series(w, index=valid.index)
            w = self._apply_cap(w)
        else:
            longs  = valid[valid >= 0]
            shorts = valid[valid <  0]
            w_long  = pd.Series(0.0, index=valid.index)
            w_short = pd.Series(0.0, index=valid.index)
            if len(longs) > 0:
                w_long[longs.index]   =  scipy_softmax(longs.values * self.temperature)
            if len(shorts) > 0:
                w_short[shorts.index] = -scipy_softmax(np.abs(shorts.values) * self.temperature)
            w = self._apply_cap(w_long + w_short)

        return w.reindex(signal_row.index).fillna(0.0)

    def _apply_cap(self, w: pd.Series) -> pd.Series:
        for _ in range(10):
            capped = w.clip(upper=self.max_weight)
            total  = capped.sum()
            if total <= 0:
                break
            capped /= total
            if (capped <= self.max_weight + 1e-9).all():
                return capped
            w = capped
        return capped

    def _rebalance_dates(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Calcule les dates de rebalancement comme la DERNIÈRE date boursière
        réelle de chaque période (semaine / mois / jour).
        """
        freq_map = {
            "D": "D",
            "W": "W-FRI",   # regroupe lundi→vendredi, fin = vendredi
            "M": "ME",       # fin de mois calendaire
        }
        grouper_freq = freq_map.get(self.rebalance_freq, self.rebalance_freq)

        # Série temporaire pour pouvoir utiliser groupby(Grouper)
        tmp = pd.Series(index=index, data=range(len(index)), dtype=int)
        groups = tmp.groupby(pd.Grouper(freq=grouper_freq))

        # Dernière date RÉELLE de chaque groupe (= dernière date boursière)
        rebal = pd.DatetimeIndex(sorted([
            grp.index[-1] for _, grp in groups if len(grp) > 0
        ]))
        return rebal

    def compute_weights(self, signal: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les poids softmax à chaque date de rebalancement
        puis forward-fill jusqu'au prochain rebalancement.
        """
        rebal_dates = self._rebalance_dates(signal.index)

        weights_rebal = pd.DataFrame(
            index=rebal_dates, columns=signal.columns, dtype=float
        )

        for date in rebal_dates:
            row = signal.loc[date]
            weights_rebal.loc[date] = self._softmax_weights(row)

        # Forward-fill sur l'index complet du signal
        weights = weights_rebal.reindex(signal.index).ffill().fillna(0.0)

        # Diagnostic
        avg_n      = (weights > 0.001).sum(axis=1).mean()
        avg_max_w  = weights.max(axis=1).mean()
        avg_hhi    = (weights ** 2).sum(axis=1).mean()
        n          = len(signal.columns)

        print(
            f"[Portfolio Softmax] {len(rebal_dates)} rebalancements\n"
            f"  Température       : {self.temperature}\n"
            f"  Plafond/actif     : {self.max_weight:.0%}\n"
            f"  Actifs actifs moy : {avg_n:.1f}/{n}\n"
            f"  Poids max moyen   : {avg_max_w:.1%}\n"
            f"  HHI moyen         : {avg_hhi:.4f}  "
            f"(équipondéré = {1/n:.4f})"
        )
        return weights

    @staticmethod
    def equal_weight(tickers: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(1.0 / len(tickers), index=dates, columns=tickers)

    def diversification_report(self, weights: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "n_active"   : (weights > 0.01).sum(axis=1),
            "max_weight" : weights.max(axis=1),
            "hhi"        : (weights ** 2).sum(axis=1),
        })


if __name__ == "__main__":
    from data.loader import download_prices
    prices, _ = download_prices()
    tickers   = list(prices.columns)
    rng       = np.random.default_rng(0)
    signal    = pd.DataFrame(
        rng.standard_normal((len(prices), len(tickers))),
        index=prices.index, columns=tickers,
    )
    signal = signal.rank(axis=1, pct=True) * 2 - 1
    pc = PortfolioConstructor(temperature=2.0, max_weight=0.15, rebalance_freq="W")
    w  = pc.compute_weights(signal)
    print(w.tail().round(4))
    print("Somme :", w.iloc[-1].sum().round(4))