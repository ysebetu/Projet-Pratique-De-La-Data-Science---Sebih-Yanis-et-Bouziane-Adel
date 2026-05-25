# models/supervised.py
# ─────────────────────────────────────────────────────────────────────────────
# XGBoostPanelSignal — Modèle cross-asset sur panel data (pooled)
#
#   Au lieu d'entraîner un modèle XGBoost séparé par action (approche
#   time-series par asset), on construit un seul modèle global entraîné
#   sur TOUTES les actions simultanément.
#
#   Chaque observation est un couple (date, ticker) avec :
#     - features ticker-level  : rendement, momentum, volatilité propres à l'action
#     - features market-level  : rendement moyen du marché, volatilité du marché
#     - ticker encoding        : one-hot pour capturer les effets fixes par action
#     - target                 : rendement J+1 de cette action à cette date
#
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from config import XGBOOST_PARAMS


# Features propres à chaque action (ticker-level)
TICKER_FEATURES = ["ret", "mom5", "mom20", "vol20"]


class XGBoostPanelSignal:
    """
    Modèle XGBoost unique entraîné sur un panel (date × ticker).

    Construction du dataset :
    ─────────────────────────
    Pour chaque date t et chaque ticker i, une ligne du dataset est :

        [ret_i,t | mom5_i,t | mom20_i,t | vol20_i,t |   ← ticker-level features
         mkt_ret_t | mkt_vol_t | mkt_mom5_t |           ← market-level features
         onehot_ticker_i]                                ← ticker fixed effect
        → target : ret_i,t+1

    Walk-forward global :
    ─────────────────────
    Un seul split temporel strict (pas de split par ticker).
    - Train : dates 0 → split_date  (toutes les actions)
    - Test  : dates split_date → fin (toutes les actions)

    Le modèle apprend ainsi les patterns cross-sectionnels : il peut comparer
    les caractéristiques de deux actions à la même date et apprendre que
    l'action avec le momentum le plus fort sur fond de marché haussier
    aura tendance à surperformer.
    """

    def __init__(
        self,
        params: dict       = XGBOOST_PARAMS,
        train_ratio: float = 0.70,    # 70% pour l'entraînement
        val_ratio: float   = 0.10,    # 10% pour la validation (early stopping)
    ):
        self.params      = params
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.model       = None
        self.scaler      = StandardScaler()
        self.tickers_    : list[str]  = []
        self.feature_names_: list[str] = []
        self.split_date_ : pd.Timestamp | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Construction du panel (long format)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_panel(
        self,
        asset_features: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construit le dataset panel (date × ticker) sans aucun fillna.

        Paramètres
        ----------
        asset_features : dict ticker → DataFrame(ret, mom5, mom20, vol20, target)
                         tel que retourné par build_asset_features()
        returns        : DataFrame (dates × tickers) des rendements log

        Retourne
        --------
        panel : DataFrame avec MultiIndex (date, ticker)
                colonnes : TICKER_FEATURES + market features + one-hot tickers + target
        """
        # ── Features market-level (agrégées sur l'ensemble des tickers) ──────
        # Calculées sur l'index complet des rendements pour avoir la couverture max
        mkt_ret  = returns.mean(axis=1).rename("mkt_ret")
        mkt_vol  = returns.std(axis=1).rename("mkt_vol")
        mkt_mom5 = mkt_ret.rolling(5).mean().rename("mkt_mom5")
        market_df = pd.concat([mkt_ret, mkt_vol, mkt_mom5], axis=1)

        # ── One-hot encoding des tickers ──────────────────────────────────────
        # Permet au modèle d'apprendre des effets fixes par action :
        # par exemple, LVMH a structurellement un beta plus fort que BNP.
        tickers = sorted(asset_features.keys())
        self.tickers_ = tickers
        onehot = pd.get_dummies(tickers, prefix="ticker").set_index(
            pd.Index(tickers, name="ticker")
        ).astype(float)

        # ── Normalisation du nom d'index avant jointure ──────────────────────
        # yfinance peut renvoyer "Date" (majuscule) selon la version installée
        market_df.index.name = "date"

        # ── Assemblage du panel ───────────────────────────────────────────────
        frames = []
        for ticker in tickers:
            df_t = asset_features[ticker][TICKER_FEATURES + ["target"]].copy()
            # Forcer le nom de l'index en minuscule pour cohérence
            df_t.index.name = "date"

            # Jointure avec les features de marché (sur les dates communes)
            df_t = df_t.join(market_df, how="inner")

            # Ajout de la colonne ticker pour le one-hot
            df_t["ticker"] = ticker

            frames.append(df_t)

        # Concaténation de toutes les actions → panel long format
        panel = pd.concat(frames, axis=0)

        panel = panel.reset_index()
        panel = panel.rename(columns={
            c: "date" for c in panel.columns
            if c.lower() == "date" and c != "date"
        })
        if "date" not in panel.columns:
            # Dernier recours : la première colonne est l'index
            panel = panel.rename(columns={panel.columns[0]: "date"})

        panel = panel.set_index(["date", "ticker"])

        # Jointure du one-hot encoding
        panel = panel.join(onehot, on="ticker")

        # ── Nettoyage strict : aucun NaN autorisé ─────────────────────────────
        n_before = len(panel)
        panel = panel.dropna()
        n_after = len(panel)
        if n_before > n_after:
            print(f"[Panel] {n_before - n_after} lignes supprimées (NaN) → {n_after} observations")

        # Trier par date puis par ticker pour la cohérence du split temporel
        panel = panel.sort_index(level="date")

        self.feature_names_ = [c for c in panel.columns if c != "target"]
        print(
            f"[Panel] Dataset construit : {n_after:,} observations | "
            f"{len(self.feature_names_)} features | "
            f"{len(tickers)} tickers"
        )
        return panel

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Split temporel strict
    # ─────────────────────────────────────────────────────────────────────────
    def _temporal_split(self, panel: pd.DataFrame) -> tuple:
        """
        Découpe le panel en train / validation / test par date.

        IMPORTANT : le split est uniquement sur la dimension temporelle.
        On ne mélange jamais des données futures avec des données passées,
        même entre tickers différents. 

        Retourne (X_train, y_train, X_val, y_val, X_test, y_test, test_dates)
        """
        dates_unique = panel.index.get_level_values("date").unique().sort_values()
        n = len(dates_unique)

        n_train = int(n * self.train_ratio)
        n_val   = int(n * (self.train_ratio + self.val_ratio))

        train_end = dates_unique[n_train - 1]
        val_end   = dates_unique[n_val   - 1]
        self.split_date_ = train_end

        mask_train = panel.index.get_level_values("date") <= train_end
        mask_val   = (panel.index.get_level_values("date") > train_end) & \
                     (panel.index.get_level_values("date") <= val_end)
        mask_test  = panel.index.get_level_values("date") > val_end

        def split(mask):
            sub = panel.loc[mask]
            return sub[self.feature_names_].values, sub["target"].values, sub.index

        X_tr, y_tr, idx_tr = split(mask_train)
        X_va, y_va, idx_va = split(mask_val)
        X_te, y_te, idx_te = split(mask_test)

        print(
            f"[Panel] Split temporel strict :\n"
            f"  Train : jusqu'au {train_end.date()} → {len(y_tr):,} obs\n"
            f"  Val   : {train_end.date()} → {val_end.date()} → {len(y_va):,} obs\n"
            f"  Test  : {val_end.date()} → fin → {len(y_te):,} obs"
        )
        return X_tr, y_tr, X_va, y_va, X_te, y_te, idx_te

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Entraînement
    # ─────────────────────────────────────────────────────────────────────────
    def fit(
        self,
        asset_features: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
    ) -> None:
        """
        Construit le panel, fait le split temporel, standardise les features,
        et entraîne le modèle XGBoost global.
        """
        panel = self._build_panel(asset_features, returns)
        X_tr, y_tr, X_va, y_va, X_te, y_te, _ = self._temporal_split(panel)

        # Standardisation des features numériques 
        # On standardise sur le train uniquement pour éviter le data leakage
        n_onehot = len(self.tickers_)
        n_numeric = len(self.feature_names_) - n_onehot

        # Scaler appliqué uniquement aux colonnes numériques
        X_tr_num = self.scaler.fit_transform(X_tr[:, :n_numeric])
        X_va_num = self.scaler.transform(X_va[:, :n_numeric])

        X_tr_scaled = np.hstack([X_tr_num, X_tr[:, n_numeric:]])
        X_va_scaled = np.hstack([X_va_num, X_va[:, n_numeric:]])

        # ── Entraînement XGBoost avec early stopping sur la validation ────────
        self.model = XGBRegressor(
            **self.params,
            verbosity=0,
            early_stopping_rounds=20,
            eval_metric="rmse",
        )
        self.model.fit(
            X_tr_scaled, y_tr,
            eval_set=[(X_va_scaled, y_va)],
            verbose=False,
        )
        self._n_numeric = n_numeric

        # Importance des features (informatif)
        importances = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names_,
        ).sort_values(ascending=False)
        print(f"\n[XGBoost] Top-5 features importantes :")
        print(importances.head(5).to_string())

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Génération du signal
    # ─────────────────────────────────────────────────────────────────────────
    def generate_signal(
        self,
        asset_features: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Entraîne le modèle si ce n'est pas déjà fait, puis génère les
        prédictions pour toute la période de test (hors-échantillon strict).

        Retourne un DataFrame signal (dates × tickers) normalisé en rang
        cross-sectionnel dans [-1, 1], compatible avec le reste du pipeline.
        """
        # ── Construction du panel complet ─────────────────────────────────────
        panel = self._build_panel(asset_features, returns)

        if self.model is None:
            X_tr, y_tr, X_va, y_va, X_te, y_te, idx_te = self._temporal_split(panel)
            n_numeric = len(self.feature_names_) - len(self.tickers_)
            X_tr_num = self.scaler.fit_transform(X_tr[:, :n_numeric])
            X_va_num = self.scaler.transform(X_va[:, :n_numeric])
            X_tr_s = np.hstack([X_tr_num, X_tr[:, n_numeric:]])
            X_va_s = np.hstack([X_va_num, X_va[:, n_numeric:]])
            self.model = XGBRegressor(
                **self.params, verbosity=0,
                early_stopping_rounds=20, eval_metric="rmse",
            )
            self.model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)
            self._n_numeric = n_numeric
        else:
            _, _, _, _, X_te, y_te, idx_te = self._temporal_split(panel)

        # ── Prédictions sur l'ensemble du test ───────────────────────────────
        X_te_all = panel[self.feature_names_].values
        X_te_num = self.scaler.transform(X_te_all[:, :self._n_numeric])
        X_te_scaled = np.hstack([X_te_num, X_te_all[:, self._n_numeric:]])

        preds_all = self.model.predict(X_te_scaled)

        # Reconstruction DataFrame (date × ticker)
        preds_series = pd.Series(preds_all, index=panel.index, name="pred")
        signal_wide  = preds_series.unstack("ticker")   # dates × tickers

        # ── Normalisation en rang cross-sectionnel → [-1, 1] ─────────────────
        # À chaque date, on compare toutes les actions entre elles.
        # L'action avec la meilleure prédiction reçoit le score +1,
        # la pire reçoit -1. C'est la normalisation standard en quant finance.
        ranked = signal_wide.rank(axis=1, pct=True) * 2 - 1

        # Masque hors-échantillon : on ne renvoie que les prédictions test
        # pour éviter toute contamination dans l'évaluation du pipeline.
        if self.split_date_ is not None:
            ranked = ranked.loc[ranked.index > self.split_date_]

        print(
            f"[XGBoost] Signal généré : {len(ranked)} dates × "
            f"{len(ranked.columns)} tickers (hors-échantillon)"
        )
        return ranked

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Rapport de feature importance
    # ─────────────────────────────────────────────────────────────────────────
    def feature_importance_report(self) -> pd.Series:
        """Retourne l'importance normalisée de chaque feature."""
        if self.model is None:
            raise RuntimeError("Modèle non entraîné.")
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names_,
        ).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Test standalone
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data.loader import download_prices
    from features.build_features import build_asset_features, compute_returns

    prices, _ = download_prices()
    returns   = compute_returns(prices)
    asset_feats = build_asset_features(prices)

    model = XGBoostPanelSignal(train_ratio=0.70, val_ratio=0.10)
    model.fit(asset_feats, returns)
    sig = model.generate_signal(asset_feats, returns)

    print("\nSignal (5 dernières dates) :")
    print(sig.tail())
    print("\nFeature importance :")
    print(model.feature_importance_report().to_string())
