# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Configuration centrale du projet
# ─────────────────────────────────────────────────────────────────────────────

# ── Univers d'investissement : CAC 40 complet ─────────────────────────────────
CAC40_TICKERS = [
    "AC.PA",      # Accor
    "AI.PA",      # Air Liquide
    "AIR.PA",     # Airbus
    "MT.AS",      # ArcelorMittal
    "CS.PA",      # AXA
    "BNP.PA",     # BNP Paribas
    "EN.PA",      # Bouygues
    "BVI.PA",     # Bureau Veritas
    "CAP.PA",     # Capgemini
    "CA.PA",      # Carrefour
    "ACA.PA",     # Crédit Agricole
    "BN.PA",      # Danone
    "DSY.PA",     # Dassault Systèmes
    "FGR.PA",     # Eiffage
    "ENGI.PA",    # Engie
    "EL.PA",      # EssilorLuxottica
    "ERF.PA",     # Eurofins Scientific
    "RMS.PA",     # Hermès
    "KER.PA",     # Kering
    "OR.PA",      # L'Oréal
    "LR.PA",      # Legrand
    "MC.PA",      # LVMH
    "ML.PA",      # Michelin
    "ORA.PA",     # Orange
    "RI.PA",      # Pernod Ricard
    "PUB.PA",     # Publicis
    "RNO.PA",     # Renault
    "SAF.PA",     # Safran
    "SGO.PA",     # Saint-Gobain
    "SAN.PA",     # Sanofi
    "SU.PA",      # Schneider Electric
    "GLE.PA",     # Société Générale
    "STMPA.PA",   # STMicroelectronics
    "HO.PA",      # Thales
    "TTE.PA",     # TotalEnergies
    "URW.PA",     # Unibail-Rodamco-Westfield
    "VIE.PA",     # Veolia
    "DG.PA",      # Vinci
    "WLN.PA",     # Worldline
    "TEP.PA",     # Teleperformance
]

# Déduplication au cas où
CAC40_TICKERS = list(dict.fromkeys(CAC40_TICKERS))

# Benchmark CAC 40
BENCHMARK_TICKER = "^FCHI"

# ── Période de données ────────────────────────────────────────────────────────
START_DATE = "2021-01-01"
END_DATE   = "2026-05-01"

# ── Features ──────────────────────────────────────────────────────────────────
MOMENTUM_WINDOWS   = [5, 20]
VOLATILITY_WINDOW  = 20

# ── Modèles ───────────────────────────────────────────────────────────────────
N_CLUSTERS      = 3
XGBOOST_PARAMS  = {
    "n_estimators"    : 300,
    "max_depth"       : 4,
    "learning_rate"   : 0.03,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state"    : 42,
}
LSTM_LOOKBACK   = 20
LSTM_EPOCHS     = 30
LSTM_BATCH_SIZE = 32

# ── Signaux & combinaison ─────────────────────────────────────────────────────
RIDGE_ALPHA = 1.0

# ── Portefeuille ──────────────────────────────────────────────────────────────
# Softmax : tous les actifs reçoivent un poids, proportionnel à leur score
# SOFTMAX_TEMPERATURE contrôle la concentration :
#   - valeur élevée (ex: 5.0)  → concentration forte sur les meilleurs
SOFTMAX_TEMPERATURE  = 5.0
# Plafond de poids par actif (évite la sur-concentration)
MAX_WEIGHT_PER_ASSET = 0.20        # 20% max par action

TRANSACTION_COST = 0.001           # 10 bp 
REBALANCE_FREQ   = "W"             # hebdomadaire

# ── Backtest ──────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000

# ── Chemins ───────────────────────────────────────────────────────────────────
DATA_PATH    = "data/prices.parquet"
RESULTS_PATH = "results/"
