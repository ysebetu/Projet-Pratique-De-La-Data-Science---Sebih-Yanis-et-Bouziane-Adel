# data/loader.py
# ─────────────────────────────────────────────────────────────────────────────
# Téléchargement et mise en cache des prix ajustés (yfinance)
# ─────────────────────────────────────────────────────────────────────────────

import os
import yfinance as yf
import pandas as pd
from config import CAC40_TICKERS, BENCHMARK_TICKER, START_DATE, END_DATE, DATA_PATH

PRICES_PATH    = DATA_PATH
BENCHMARK_PATH = DATA_PATH.replace(".parquet", "_benchmark.parquet")


def download_prices(
    tickers       : list[str] = CAC40_TICKERS,
    benchmark     : str       = BENCHMARK_TICKER,
    start         : str       = START_DATE,
    end           : str       = END_DATE,
    force_reload  : bool      = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    --------
    prices    : DataFrame  (dates × tickers)   — NaN forward-fillés intra-ticker
    benchmark : Series     (dates)              — cours du CAC 40
    """
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    # ── Actions ───────────────────────────────────────────────────────────────
    if not force_reload and os.path.exists(PRICES_PATH):
        print(f"[Loader] Chargement cache actions : {PRICES_PATH}")
        prices_df = pd.read_parquet(PRICES_PATH)
    else:
        print(f"[Loader] Téléchargement actions ({start} → {end}) …")
        raw = yf.download(
            tickers,
            start        = start,
            end          = end,
            auto_adjust  = True,
            progress     = True,
        )["Close"]

        if isinstance(raw, pd.Series):
            raw = raw.to_frame()

        # Supprimer les colonnes entièrement vides (ticker introuvable)
        raw = raw.dropna(axis=1, how="all")

        # Forward-fill les NaN intra-colonne (jours fériés locaux)
        # puis supprimer les lignes encore vides sur TOUTES les colonnes
        raw = raw.ffill().dropna(axis=0, how="all")

        prices_df = raw
        prices_df.to_parquet(PRICES_PATH)
        print(f"[Loader] Actions sauvegardées → {PRICES_PATH}")

    # ── Benchmark (séparé pour ne pas perdre de dates) ────────────────────────
    if not force_reload and os.path.exists(BENCHMARK_PATH):
        print(f"[Loader] Chargement cache benchmark : {BENCHMARK_PATH}")
        bench_df = pd.read_parquet(BENCHMARK_PATH)
        bench_s  = bench_df.iloc[:, 0]
    else:
        print(f"[Loader] Téléchargement benchmark {benchmark} …")
        raw_b = yf.download(
            benchmark,
            start       = start,
            end         = end,
            auto_adjust = True,
            progress    = False,
        )["Close"]

        if isinstance(raw_b, pd.DataFrame):
            raw_b = raw_b.iloc[:, 0]

        bench_s = raw_b.ffill().dropna()
        bench_s.name = benchmark
        bench_s.to_frame().to_parquet(BENCHMARK_PATH)
        print(f"[Loader] Benchmark sauvegardé → {BENCHMARK_PATH}")

    # ── Nettoyage final ───────────────────────────────────────────────────────
    available = [t for t in tickers if t in prices_df.columns]
    missing   = set(tickers) - set(available)
    if missing:
        print(f"[Loader] Tickers absents : {missing}")

    prices = prices_df[available].copy()
    prices.index.name = "date"
    bench_s.index.name = "date"

    print(f"[Loader] {len(prices)} dates | {len(available)} actifs | "
          f"benchmark : {len(bench_s)} dates")

    return prices, bench_s


if __name__ == "__main__":
    prices, bench = download_prices(force_reload=True)
    print(prices.tail())
    print(bench.tail())
