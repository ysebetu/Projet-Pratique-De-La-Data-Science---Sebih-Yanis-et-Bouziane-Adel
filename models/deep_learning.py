# models/deep_learning.py
# ─────────────────────────────────────────────────────────────────────────────
# LSTMPanelSignal — Un seul LSTM global entraîné sur toutes les actions
#
#   Au lieu de 40 LSTMs séparés, on entraîne UN SEUL modèle sur un panel
#   de séquences provenant de toutes les actions.
#
#   Chaque observation d'entraînement est une séquence de L=20 jours pour
#   un couple (date, ticker) donné :
#
#       Input  : séquence [t-19, …, t] × 7 features
#                  - 4 features ticker-level : ret, mom5, mom20, vol20
#                  - 3 features market-level : mkt_ret, mkt_vol, mkt_mom5
#       Output : rendement r_{i,t+1}
#
#   Le ticker est encodé via un Embedding layer appris — plus riche qu'un
#   one-hot car il projette chaque action dans un espace continu de dimension
#   EMBED_DIM, permettant au modèle d'apprendre des similarités structurelles
#   entre actions (ex : BNP et GLE proches car toutes deux cycliques bancaires).
#
# ARCHITECTURE :
#
#   [Séquence 20j × 7 features] → LSTM(64) → LSTM(32)
#                                                    ↘
#   [ticker_id]  → Embedding(n_tickers, EMBED_DIM) → Concat → Dense(16) → Dense(1)
#
# SPLIT TEMPOREL :
#   Identique à XGBoostPanelSignal : 70% train / 10% val / 20% test
#   défini sur la dimension temporelle uniquement.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from config import LSTM_LOOKBACK, LSTM_EPOCHS, LSTM_BATCH_SIZE

# Dimension de l'embedding par ticker
EMBED_DIM   = 8
# Features continues (ticker-level + market-level)
SEQ_FEATURES = ["ret", "mom5", "mom20", "vol20", "mkt_ret", "mkt_vol", "mkt_mom5"]

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, LSTM, Dense, Dropout, Embedding, Flatten, Concatenate
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
    # Désactiver les logs verbeux TF sauf erreurs
    tf.get_logger().setLevel("ERROR")
except ImportError:
    TF_AVAILABLE = False
    print("[LSTM] TensorFlow non disponible — signal LSTM mis à zéro.")


class LSTMPanelSignal:
    """
    LSTM unique entraîné sur le panel complet (toutes actions, toutes dates).

    Paramètres
    ----------
    lookback    : longueur de la fenêtre temporelle (défaut 20 jours)
    epochs      : nombre d'epochs max (early stopping actif)
    batch_size  : taille des mini-batches
    train_ratio : fraction de l'historique pour l'entraînement
    val_ratio   : fraction pour la validation (early stopping)
    embed_dim   : dimension de l'embedding ticker
    """

    def __init__(
        self,
        lookback    : int   = LSTM_LOOKBACK,
        epochs      : int   = LSTM_EPOCHS,
        batch_size  : int   = LSTM_BATCH_SIZE,
        train_ratio : float = 0.70,
        val_ratio   : float = 0.10,
        embed_dim   : int   = EMBED_DIM,
    ):
        self.lookback    = lookback
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.embed_dim   = embed_dim

        self.model       = None
        self.scaler      = StandardScaler()
        self.ticker_map_ : dict[str, int] = {}   # ticker → entier pour l'embedding
        self.split_date_ : pd.Timestamp | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Construction du panel de séquences
    # ─────────────────────────────────────────────────────────────────────────
    def _build_sequences(
        self,
        asset_features : dict[str, pd.DataFrame],
        returns        : pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
        """
        Construit les séquences LSTM pour toutes les actions.

        Pour chaque action i et chaque date t (avec t >= lookback) :
          - X_seq[n]    : array (lookback × len(SEQ_FEATURES)) — séquence continue
          - X_ticker[n] : entier — identifiant de l'action (pour l'embedding)
          - y[n]        : float — rendement r_{i,t+1}
          - dates[n]    : date t (pour le split et la reconstruction du signal)
          - tickers[n]  : ticker i

        Retourne (X_seq, X_ticker, y, dates_arr, tickers_arr)
        """
        # ── Features market-level ──────────────────────────────────────────────
        mkt_ret  = returns.mean(axis=1).rename("mkt_ret")
        mkt_vol  = returns.std(axis=1).rename("mkt_vol")
        mkt_mom5 = mkt_ret.rolling(5).mean().rename("mkt_mom5")
        market_df = pd.concat([mkt_ret, mkt_vol, mkt_mom5], axis=1)
        market_df.index.name = "date"

        # ── Mapping ticker → entier ────────────────────────────────────────────
        tickers_sorted = sorted(asset_features.keys())
        self.ticker_map_ = {t: i for i, t in enumerate(tickers_sorted)}

        # ── Assemblage toutes actions ──────────────────────────────────────────
        all_X_seq    = []
        all_X_ticker = []
        all_y        = []
        all_dates    = []
        all_tickers  = []

        for ticker in tickers_sorted:
            df_t = asset_features[ticker][["ret", "mom5", "mom20", "vol20", "target"]].copy()
            df_t.index.name = "date"

            # Jointure avec features de marché
            df_t = df_t.join(market_df[["mkt_ret", "mkt_vol", "mkt_mom5"]], how="inner")
            df_t = df_t.dropna()

            if len(df_t) < self.lookback + 1:
                continue   # pas assez de données pour ce ticker

            feat_values = df_t[SEQ_FEATURES].values   # (T × 7)
            targets     = df_t["target"].values         # (T,)
            dates_idx   = df_t.index                    # DatetimeIndex

            ticker_id   = self.ticker_map_[ticker]

            # Découpage en séquences glissantes
            for i in range(self.lookback, len(df_t)):
                seq = feat_values[i - self.lookback : i]   # (lookback × 7)
                all_X_seq.append(seq)
                all_X_ticker.append(ticker_id)
                all_y.append(targets[i])
                all_dates.append(dates_idx[i])
                all_tickers.append(ticker)

        X_seq    = np.array(all_X_seq,    dtype=np.float32)   # (N, lookback, 7)
        X_ticker = np.array(all_X_ticker, dtype=np.int32)      # (N,)
        y        = np.array(all_y,        dtype=np.float32)    # (N,)
        dates    = np.array(all_dates)
        tickers  = np.array(all_tickers)

        print(
            f"[LSTM Panel] Séquences construites : {len(y):,} obs | "
            f"{len(tickers_sorted)} tickers | fenêtre {self.lookback}j"
        )
        return X_seq, X_ticker, y, dates, tickers

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Split temporel strict
    # ─────────────────────────────────────────────────────────────────────────
    def _temporal_split(self, dates: np.ndarray):
        """
        Split par date (même logique que XGBoostPanelSignal).
        Retourne les masques booléens train / val / test.
        """
        unique_dates = np.unique(dates)
        unique_dates.sort()
        n = len(unique_dates)

        n_train = int(n * self.train_ratio)
        n_val   = int(n * (self.train_ratio + self.val_ratio))

        train_end = unique_dates[n_train - 1]
        val_end   = unique_dates[n_val   - 1]
        self.split_date_ = pd.Timestamp(train_end)

        mask_train = dates <= train_end
        mask_val   = (dates > train_end) & (dates <= val_end)
        mask_test  = dates > val_end

        print(
            f"[LSTM Panel] Split temporel :\n"
            f"  Train : jusqu'au {pd.Timestamp(train_end).date()} → {mask_train.sum():,} séq.\n"
            f"  Val   : → {pd.Timestamp(val_end).date()} → {mask_val.sum():,} séq.\n"
            f"  Test  : → fin → {mask_test.sum():,} séq."
        )
        return mask_train, mask_val, mask_test

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Construction du modèle Keras
    # ─────────────────────────────────────────────────────────────────────────
    def _build_model(self, n_tickers: int) -> "tf.keras.Model":
        """
        Architecture dual-input :
          - Branche séquence : LSTM(64) → Dropout → LSTM(32) → Dropout
          - Branche ticker   : Embedding(n_tickers, embed_dim) → Flatten
          - Fusion           : Concatenate → Dense(16, relu) → Dense(1)
        """
        # Branche 1 : séquence temporelle
        seq_input  = Input(shape=(self.lookback, len(SEQ_FEATURES)), name="sequence")
        x = LSTM(64, return_sequences=True, name="lstm_1")(seq_input)
        x = Dropout(0.2)(x)
        x = LSTM(32, return_sequences=False, name="lstm_2")(x)
        x = Dropout(0.2)(x)

        # Branche 2 : embedding ticker
        ticker_input = Input(shape=(1,), name="ticker_id")
        emb = Embedding(
            input_dim=n_tickers,
            output_dim=self.embed_dim,
            name="ticker_embedding"
        )(ticker_input)
        emb = Flatten()(emb)

        # Fusion des deux branches
        merged = Concatenate()([x, emb])
        out    = Dense(16, activation="relu")(merged)
        out    = Dense(1,  activation="linear", name="output")(out)

        model = Model(inputs=[seq_input, ticker_input], outputs=out)
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"],
        )
        return model

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Standardisation (features continues uniquement)
    # ─────────────────────────────────────────────────────────────────────────
    def _scale(self, X_seq: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Standardise les features continues (les 7 colonnes de chaque pas de temps).
        Le scaler est ajusté sur le train uniquement (fit=True),
        puis appliqué sans ré-ajustement sur val/test (fit=False).
        """
        N, L, F = X_seq.shape
        X_flat = X_seq.reshape(-1, F)   # (N*L, F)
        if fit:
            X_scaled = self.scaler.fit_transform(X_flat)
        else:
            X_scaled = self.scaler.transform(X_flat)
        return X_scaled.reshape(N, L, F).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Entraînement
    # ─────────────────────────────────────────────────────────────────────────
    def fit(
        self,
        asset_features : dict[str, pd.DataFrame],
        returns        : pd.DataFrame,
    ) -> None:
        """
        Construit les séquences, fait le split, standardise, entraîne le LSTM.
        """
        if not TF_AVAILABLE:
            print("[LSTM] TensorFlow absent — fit() ignoré.")
            return

        X_seq, X_ticker, y, dates, tickers = self._build_sequences(asset_features, returns)
        mask_train, mask_val, _ = self._temporal_split(dates)

        # Standardisation sur le train uniquement
        X_train_s = self._scale(X_seq[mask_train], fit=True)
        X_val_s   = self._scale(X_seq[mask_val],   fit=False)

        n_tickers = len(self.ticker_map_)
        self.model = self._build_model(n_tickers)

        print(f"\n[LSTM Panel] Architecture :")
        self.model.summary(print_fn=lambda s: print("  " + s))

        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=5,
                restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=3, min_lr=1e-5, verbose=0
            ),
        ]

        print(f"\n[LSTM Panel] Entraînement ({mask_train.sum():,} séquences)…")
        self.model.fit(
            [X_train_s, X_ticker[mask_train]],
            y[mask_train],
            validation_data=(
                [X_val_s, X_ticker[mask_val]],
                y[mask_val],
            ),
            epochs     = self.epochs,
            batch_size = self.batch_size,
            callbacks  = callbacks,
            verbose    = 1,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Génération du signal
    # ─────────────────────────────────────────────────────────────────────────
    def generate_signal(
        self,
        asset_features : dict[str, pd.DataFrame],
        returns        : pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Entraîne si nécessaire, puis génère les prédictions hors-échantillon.

        Retourne un DataFrame signal (dates × tickers) normalisé en rang
        cross-sectionnel dans [-1, 1], compatible avec le pipeline existant.
        """
        if not TF_AVAILABLE:
            # Signal nul si TF absent — le pipeline continue sans planter
            all_dates   = pd.concat(asset_features.values()).index.unique().sort_values()
            tickers_lst = sorted(asset_features.keys())
            return pd.DataFrame(0.0, index=all_dates, columns=tickers_lst)

        # Entraîner si pas encore fait
        if self.model is None:
            self.fit(asset_features, returns)

        # Reconstruire toutes les séquences pour la prédiction
        X_seq, X_ticker, y, dates, tickers_arr = self._build_sequences(
            asset_features, returns
        )
        _, _, mask_test = self._temporal_split(dates)

        # Prédire sur TOUTE la période test
        X_test_s = self._scale(X_seq[mask_test], fit=False)
        preds    = self.model.predict(
            [X_test_s, X_ticker[mask_test]],
            batch_size = self.batch_size,
            verbose    = 0,
        ).flatten()

        # Reconstruction DataFrame (date × ticker)
        df_pred = pd.DataFrame({
            "date"   : pd.to_datetime(dates[mask_test]),
            "ticker" : tickers_arr[mask_test],
            "pred"   : preds,
        })
        signal_wide = df_pred.pivot_table(
            index="date", columns="ticker", values="pred", aggfunc="mean"
        )

        # Normalisation en rang cross-sectionnel → [-1, 1]
        ranked = signal_wide.rank(axis=1, pct=True) * 2 - 1

        print(
            f"[LSTM Panel] Signal généré : {len(ranked)} dates × "
            f"{len(ranked.columns)} tickers (hors-échantillon)"
        )
        return ranked


# ─────────────────────────────────────────────────────────────────────────────
# Test 
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data.loader import download_prices
    from features.build_features import build_asset_features, compute_returns

    prices, _ = download_prices()
    returns   = compute_returns(prices)
    asset_feats = build_asset_features(prices)

    model = LSTMPanelSignal(epochs=5, lookback=20)   # epochs=5 pour test rapide
    sig   = model.generate_signal(asset_feats, returns)
    print("\nSignal (5 dernières dates) :")
    print(sig.tail())
