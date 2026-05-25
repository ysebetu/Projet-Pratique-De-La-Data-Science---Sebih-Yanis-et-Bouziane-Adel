# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Orchestrateur principal : lance la chaîne complète 
#   Données → Features → Modèles → Signaux → Combinaison →
#   Poids → Backtest → Évaluation → Visualisation
# ─────────────────────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")



import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
import random

SEED=42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from config import (
    CAC40_TICKERS, SOFTMAX_TEMPERATURE, MAX_WEIGHT_PER_ASSET,
    REBALANCE_FREQ, INITIAL_CAPITAL, TRANSACTION_COST, RESULTS_PATH
)
from data.loader             import download_prices
from features.build_features import compute_returns, build_asset_features
from models.unsupervised     import MarketRegimeModel
from models.supervised       import XGBoostPanelSignal
from models.deep_learning    import LSTMPanelSignal
from signals.combine         import SignalCombiner
from portfolio.weights       import PortfolioConstructor
from backtest.engine         import BacktestEngine
from utils.metrics           import performance_table, drawdown_series, rolling_sharpe


# ─────────────────────────────────────────────────────────────────────────────
# 0. Dossiers de sortie
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs("data", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Données
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 1 — Chargement des données")
print("=" * 60)

prices, benchmark  = download_prices(force_reload=True)  # True = re-télécharge et écrase le cache
returns            = compute_returns(prices)
asset_features     = build_asset_features(prices)

tickers = list(prices.columns)
dates   = prices.index


# ─────────────────────────────────────────────────────────────────────────────
# 2. Signaux individuels
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 2 — Génération des signaux")
print("=" * 60)

# ── 2a. KMeans ───────────────────────────────────────────────────────────────
print("\n[2a] Régimes de marché — KMeans")
regime_model = MarketRegimeModel()
regime_model.fit(returns)

signal_regime = regime_model.predict_signal(returns)

cluster_summary = regime_model.get_cluster_summary()

print("\n[2a] Régimes de marché — KMeans")
print("[KMeans] Régimes détectés. Momentum moyen / cluster :", cluster_summary)

# save results
pd.Series(cluster_summary).to_csv(
    os.path.join(RESULTS_PATH, "kmeans_regimes.csv")
)


# ── 2b. XGBoost Panel ────────────────────────────────────────────────────────
print("\n[2b] Prédiction — XGBoost Panel (cross-asset, panel data)")
xgb_model  = XGBoostPanelSignal(train_ratio=0.70, val_ratio=0.10)
signal_xgb = xgb_model.generate_signal(asset_features, returns)

# ── 2c. LSTM Panel ───────────────────────────────────────────────────────────
print("\n[2c] Prédiction — LSTM Panel (1 modèle global)")
lstm_model  = LSTMPanelSignal(epochs=20, lookback=20)
signal_lstm = lstm_model.generate_signal(asset_features, returns)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Combinaison des signaux — pipeline en deux phases
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 3 — Combinaison des signaux")
print("=" * 60)

# XGBoost et LSTM ne produisent des prédictions que sur leur période test
# (~20% finaux). On adopte un pipeline en 2 phases :
#
#  Phase 1 (avant les signaux ML) : portefeuille guidé par KMeans seul
#  Phase 2 (période test ML)      : Ridge combine les 3 signaux
#
# C'est la seule approche sans look-ahead bias : on n'utilise un modèle
# que sur les données qu'il n'a jamais vues.

# Début de la période où XGB ET LSTM sont tous les deux disponibles
xgb_start  = signal_xgb.index.min()  if len(signal_xgb)  > 0 else dates[-1]
lstm_start = signal_lstm.index.min() if len(signal_lstm) > 0 else dates[-1]
ml_start   = max(xgb_start, lstm_start)
print(f"Signaux ML disponibles à partir du : {ml_start.date()}")

# ── Phase 1 : KMeans seul ────────────────────────────────────────────────────
idx_phase1 = returns.index[returns.index < ml_start]
idx_phase2 = returns.index[returns.index >= ml_start]

sig_phase1 = signal_regime.reindex(index=idx_phase1, columns=tickers).fillna(0.0)

# ── Phase 2 : Ridge sur les 3 signaux ────────────────────────────────────────
signals_phase2 = {
    name: sig.reindex(index=idx_phase2, columns=tickers).ffill().fillna(0.0)
    for name, sig in {"regime": signal_regime, "xgb": signal_xgb, "lstm": signal_lstm}.items()
}

# On entraîne la Ridge sur la 1ère moitié de la phase 2
split_phase2 = idx_phase2[int(len(idx_phase2) * 0.5)]
print(f"Ridge entraînée jusqu'au : {split_phase2.date()}")

combiner   = SignalCombiner()
ret_phase2 = returns.reindex(index=idx_phase2, columns=tickers)
combiner.fit(signals_phase2, ret_phase2, train_end=split_phase2)
sig_phase2 = combiner.predict(signals_phase2)

print("\nPoids Ridge :")
print(combiner.summary().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde des poids Ridge
# ─────────────────────────────────────────────────────────────────────────────
ridge_summary = combiner.summary()

print("\nPoids Ridge :")
print(ridge_summary.to_string())

ridge_path = os.path.join(RESULTS_PATH, "ridge_weights.csv")
ridge_summary.to_csv(ridge_path)

print(f"[Results] Poids Ridge sauvegardés → {ridge_path}")

# ── Concaténation → signal complet ───────────────────────────────────────────
signal_combined = pd.concat([sig_phase1, sig_phase2], axis=0).sort_index()
common_idx      = signal_combined.index
print(f"\nIndex commun : {len(common_idx)} dates ({common_idx[0].date()} → {common_idx[-1].date()})")
print(f"  Phase 1 (KMeans seul)   : {len(idx_phase1)} dates")
print(f"  Phase 2 (Ridge 3 mod.)  : {len(idx_phase2)} dates")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Construction des portefeuilles
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 4 — Construction des portefeuilles")
print("=" * 60)

# Rendements alignés sur l'index du signal combiné (nécessaire pour le backtest)
ret_aligned = returns.reindex(index=common_idx, columns=tickers)

pc = PortfolioConstructor(
    temperature    = SOFTMAX_TEMPERATURE,
    max_weight     = MAX_WEIGHT_PER_ASSET,
    rebalance_freq = REBALANCE_FREQ,
)

weights_strategy = pc.compute_weights(signal_combined)
weights_equal    = PortfolioConstructor.equal_weight(tickers, common_idx)

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics portefeuille
# ─────────────────────────────────────────────────────────────────────────────

# Nombre moyen d'actifs investis
active_assets = (weights_strategy > 1e-4).sum(axis=1)

# Poids maximum moyen
max_weight_avg = weights_strategy.max(axis=1).mean()

# HHI = somme des poids²
hhi = (weights_strategy ** 2).sum(axis=1)

portfolio_stats = pd.DataFrame({
    "Statistique": [
        "Nombre de rebalancements",
        "Température softmax",
        "Poids max autorisé",
        "Actifs actifs moyens",
        "Poids max moyen",
        "HHI moyen",
        "HHI équipondéré théorique"
    ],
    "Valeur": [
        len(weights_strategy),
        SOFTMAX_TEMPERATURE,
        MAX_WEIGHT_PER_ASSET,
        round(active_assets.mean(), 2),
        round(max_weight_avg, 4),
        round(hhi.mean(), 4),
        round(1 / len(tickers), 4)
    ]
})

print("\n[Portfolio Softmax]")
print(portfolio_stats.to_string(index=False))

portfolio_stats_path = os.path.join(RESULTS_PATH, "portfolio_statistics.csv")
portfolio_stats.to_csv(portfolio_stats_path, index=False)

print(f"[Results] Statistiques portefeuille sauvegardées → {portfolio_stats_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Backtest
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 5 — Backtest")
print("=" * 60)

engine = BacktestEngine(
    initial_capital  = INITIAL_CAPITAL,
    transaction_cost = TRANSACTION_COST,
)

result_strategy = engine.run(weights_strategy, ret_aligned, name="Multi-Modèles")
result_equal    = engine.run(weights_equal,    ret_aligned, name="Équipondéré")

# Benchmark CAC 40 : on aligne le cours sur common_idx puis on calcule les rendements
bench_ret_aligned = (
    benchmark
    .reindex(common_idx)
    .ffill()
    .pct_change()
    .fillna(0.0)
)
result_bh = engine.run_buyhold(bench_ret_aligned, name="Buy & Hold CAC 40")

nav_df = BacktestEngine.compare_nav(
    ("Multi-Modèles",     result_strategy),
    ("Équipondéré",       result_equal),
    ("Buy & Hold CAC 40", result_bh),
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Métriques de performance
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 6 — Métriques de performance")
print("=" * 60)

bench_ret_for_ir = bench_ret_aligned.reindex(nav_df.index).fillna(0.0)

strategies_returns = {
    "Multi-Modèles"    : result_strategy["portfolio_return"].reindex(nav_df.index).fillna(0.0),
    "Équipondéré"      : result_equal["portfolio_return"].reindex(nav_df.index).fillna(0.0),
    "Buy & Hold CAC 40": result_bh["portfolio_return"].reindex(nav_df.index).fillna(0.0),
}

perf_table = performance_table(strategies_returns, bench_ret_for_ir)
print("\n" + perf_table.to_string())
perf_table.to_csv(os.path.join(RESULTS_PATH, "performance.csv"))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualisations
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 7 — Génération des graphiques")
print("=" * 60)

plt.rcParams.update({
    "figure.facecolor" : "#0d1117",
    "axes.facecolor"   : "#161b22",
    "axes.edgecolor"   : "#30363d",
    "axes.labelcolor"  : "#c9d1d9",
    "text.color"       : "#c9d1d9",
    "xtick.color"      : "#8b949e",
    "ytick.color"      : "#8b949e",
    "grid.color"       : "#21262d",
    "grid.linestyle"   : "--",
    "grid.linewidth"   : 0.5,
    "legend.framealpha": 0.3,
    "legend.edgecolor" : "#30363d",
    "font.family"      : "monospace",
})

COLORS = {
    "Multi-Modèles"    : "#58a6ff",
    "Équipondéré"      : "#3fb950",
    "Buy & Hold CAC 40": "#f78166",
}

fig, axes = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={"hspace": 0.4})

# Courbe de richesse
ax = axes[0]
for name, color in COLORS.items():
    ax.plot(nav_df.index, nav_df[name], label=name, color=color, linewidth=1.6)
ax.set_title("Courbe de richesse — Comparaison des stratégies", fontsize=13, pad=10)
ax.set_ylabel(f"NAV (capital initial = {INITIAL_CAPITAL:,} €)")
ax.legend()
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Drawdown
ax = axes[1]
for name, ret_s in strategies_returns.items():
    dd = drawdown_series(ret_s)
    ax.fill_between(dd.index, dd * 100, 0, alpha=0.4, color=COLORS[name], label=name)
    ax.plot(dd.index, dd * 100, color=COLORS[name], linewidth=0.8)
ax.set_title("Drawdown (%)", fontsize=13, pad=10)
ax.set_ylabel("Drawdown (%)")
ax.legend()
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Sharpe glissant
ax = axes[2]
for name, ret_s in strategies_returns.items():
    rs = rolling_sharpe(ret_s, window=63)
    ax.plot(rs.index, rs, label=name, color=COLORS[name], linewidth=1.2)
ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle=":")
ax.set_title("Sharpe Ratio glissant (fenêtre 63 j)", fontsize=13, pad=10)
ax.set_ylabel("Sharpe Ratio")
ax.legend()
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.suptitle(
    "Construction de portefeuilles — CAC 40 — Signaux ML/DL",
    fontsize=15, fontweight="bold", y=0.98,
)

out_path = os.path.join(RESULTS_PATH, "performance_chart.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"[Main] Graphique sauvegardé → {out_path}")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Résumé final
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RÉSUMÉ FINAL")
print("=" * 60)
print(perf_table.to_string())
print(f"\nFichiers générés dans : {RESULTS_PATH}")
print("  • ridge_weights.csv")
print("  • market_regimes.csv")
print("  • portfolio_statistics.csv")
print("  • performance.csv")
print("  • performance_chart.png")
print("\n✓ Pipeline complet terminé.")
