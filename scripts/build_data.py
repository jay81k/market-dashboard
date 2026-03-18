"""
Build stock dashboard data for static GitHub Pages deployment.

Flow:
  1. Download fresh stock universe CSV (RS-ranked list)
  2. Filter to AvgVol10 >= 100k
  3. Batch-download 1y price history from yfinance in parallel
  4. Compute metrics per stock
  5. Write data/snapshot.json + data/meta.json

Run from repo root:
  python scripts/build_data.py [--out-dir data] [--csv-url URL] [--workers 10]

GitHub Actions:
  python scripts/build_data.py --out-dir data --csv-url "$CSV_URL"
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CSV_URL = os.environ.get("CSV_URL", "")
DEFAULT_INDUSTRIES_CSV_URL = os.environ.get("INDUSTRIES_CSV_URL", "")

MIN_AVG_VOLUME = 100_000  # AvgVol10 filter — anything below gets dropped

# Columns to pass through from CSV into snapshot as-is (if present)
CSV_PASSTHROUGH = [
    "Rank", "Relative Strength", "Percentile",
    "1M_RS_Percentile", "3M_RS_Percentile", "6M_RS_Percentile",
    "MarketCap", "Float", "ShortFloatPct", "PctFrom52WkHigh",
    "AvgVol10", "AvgVol30", "AvgVol50", "RevenueGrowth", "Exchange",
]

# MA combos for Dist/MA column (matches ETF dashboard)
DIST_MA_COMBOS = [
    ("SMA", 5), ("SMA", 10), ("SMA", 21), ("SMA", 50), ("SMA", 200),
    ("EMA", 5), ("EMA", 10), ("EMA", 21), ("EMA", 50), ("EMA", 200),
]

# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------

def load_universe(csv_url: str) -> pd.DataFrame:
    """Download and validate the stock universe CSV."""
    print(f"Fetching universe CSV: {csv_url}")
    df = pd.read_csv(csv_url)

    required = {"Ticker", "Sector", "Industry"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    total_before = len(df)

    if "AvgVol10" in df.columns:
        df["AvgVol10"] = pd.to_numeric(df["AvgVol10"], errors="coerce")
        df = df[df["AvgVol10"] >= MIN_AVG_VOLUME].copy()
        removed = total_before - len(df)
        print(f"Volume filter (AvgVol10 >= {MIN_AVG_VOLUME:,}): "
              f"{total_before} → {len(df)} stocks ({removed} removed)")
    else:
        print("Warning: AvgVol10 column not found, skipping volume filter")

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df = df[df["Ticker"] != ""].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def calculate_dist_ma(close: pd.Series, ma_type: str, period: int) -> float | None:
    if len(close) < period:
        return None
    try:
        if ma_type == "EMA":
            ma = close.ewm(span=period, adjust=False).mean().iloc[-1]
        else:
            ma = close.rolling(window=period).mean().iloc[-1]
        current = close.iloc[-1]
        if ma == 0:
            return None
        return round(((current - ma) / ma) * 100, 2)
    except Exception:
        return None


def calculate_adr_pct(hist: pd.DataFrame, period: int = 20) -> float | None:
    try:
        capped = min(period, len(hist))
        adr = (hist["High"] - hist["Low"]).rolling(window=capped).mean().iloc[-1]
        current = hist["Close"].iloc[-1]
        if current and current > 0:
            return round((adr / current) * 100, 2)
    except Exception:
        pass
    return None


def compute_metrics(ticker: str, hist: pd.DataFrame, spy_hist: pd.DataFrame) -> dict | None:
    """Compute all price-derived metrics for a single ticker."""
    try:
        hist = hist.dropna(subset=["Close", "Open", "High", "Low"])
        if len(hist) < 10:
            return None

        close   = hist["Close"]
        current = close.iloc[-1]
        prev    = close.iloc[-2]

        daily       = (current / prev - 1) * 100
        one_week    = (current / close.iloc[-6]  - 1) * 100 if len(close) >= 6  else None
        one_month   = (current / close.iloc[-22] - 1) * 100 if len(close) >= 22 else None
        three_month = (current / close.iloc[-63] - 1) * 100 if len(close) >= 63 else None

        # YTD: first close of current calendar year
        year_rows = hist[hist.index.year == hist.index[-1].year]
        ytd = (current / year_rows["Close"].iloc[0] - 1) * 100 if len(year_rows) >= 1 else None

        # vs SPY relative strength (1m and 3m)
        vs_spy_1m = vs_spy_3m = None
        if spy_hist is not None:
            spy_close = spy_hist["Close"]
            if len(spy_close) >= 22 and len(close) >= 22:
                spy_1m    = (spy_close.iloc[-1] / spy_close.iloc[-22] - 1) * 100
                vs_spy_1m = round(one_month - spy_1m, 2) if one_month is not None else None
            if len(spy_close) >= 63 and len(close) >= 63:
                spy_3m    = (spy_close.iloc[-1] / spy_close.iloc[-63] - 1) * 100
                vs_spy_3m = round(three_month - spy_3m, 2) if three_month is not None else None

        # Closing range
        today_high = hist["High"].iloc[-1]
        today_low  = hist["Low"].iloc[-1]
        cr = ((current - today_low) / (today_high - today_low) * 100) \
            if (today_high - today_low) > 0 else None

        # ADR%
        adr_pct = calculate_adr_pct(hist)

        # Dist/MA for all combos
        dist_ma = {
            ma_type + str(period): calculate_dist_ma(close, ma_type, period)
            for ma_type, period in DIST_MA_COMBOS
        }

        return {
            "daily":     round(daily, 2),
            "1w":        round(one_week, 2)    if one_week    is not None else None,
            "1m":        round(one_month, 2)   if one_month   is not None else None,
            "3m":        round(three_month, 2) if three_month is not None else None,
            "ytd":       round(ytd, 2)         if ytd         is not None else None,
            "vs_spy":    vs_spy_1m,
            "vs_spy_3m": vs_spy_3m,
            "cr":        round(cr, 1)          if cr          is not None else None,
            "adr_pct":   adr_pct,
            "dist_ma":   dist_ma,
        }
    except Exception as e:
        print(f"  Metric error [{ticker}]: {e}")
        return None


# ---------------------------------------------------------------------------
# Parallel history fetching
# ---------------------------------------------------------------------------

def fetch_history_batch(tickers: list[str], max_workers: int = 10) -> dict[str, pd.DataFrame]:
    """
    Fetch 1y daily history for all tickers using yf.download() in 100-ticker chunks.
    Falls back to per-ticker ThreadPoolExecutor for any chunk that fails.
    Returns dict of ticker -> DataFrame.
    """
    results: dict[str, pd.DataFrame] = {}
    CHUNK = 100

    chunks = [tickers[i:i+CHUNK] for i in range(0, len(tickers), CHUNK)]
    print(f"Downloading history for {len(tickers)} tickers "
          f"in {len(chunks)} chunks of {CHUNK}...")

    for idx, chunk in enumerate(chunks):
        print(f"  Chunk {idx+1}/{len(chunks)} ({len(chunk)} tickers)...", end=" ", flush=True)
        try:
            raw = yf.download(
                chunk,
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if len(chunk) == 1:
                df = raw.dropna(how="all")
                if len(df) >= 10:
                    results[chunk[0]] = df
                ok = 1 if len(df) >= 10 else 0
            else:
                ok = 0
                for ticker in chunk:
                    try:
                        df = raw[ticker].dropna(how="all")
                        if len(df) >= 10:
                            results[ticker] = df
                            ok += 1
                    except Exception:
                        pass
            print(f"ok={ok}/{len(chunk)}")
        except Exception as e:
            print(f"batch failed ({e}), falling back to per-ticker...")
            def _fetch_one(t: str):
                try:
                    df = yf.Ticker(t).history(period="1y")
                    return t, df if len(df) >= 10 else None
                except Exception:
                    return t, None

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(_fetch_one, t): t for t in chunk}
                ok = 0
                for fut in as_completed(futs):
                    t, df = fut.result()
                    if df is not None:
                        results[t] = df
                        ok += 1
            print(f"fallback ok={ok}/{len(chunk)}")

        time.sleep(0.5)  # polite pause between chunks

    print(f"History ready for {len(results)}/{len(tickers)} tickers\n")
    return results


# ---------------------------------------------------------------------------
# JSON sanitize
# ---------------------------------------------------------------------------

def sanitize(obj):
    """Recursively strip NaN/Inf and convert numpy scalars for JSON serialization."""
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_industries(csv_url: str) -> list:
    """Download and parse the industries RS CSV."""
    print(f"Fetching industries CSV: {csv_url}")
    df = pd.read_csv(csv_url)
    required = {"Industry", "Sector", "Percentile", "Tickers"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Industries CSV missing columns: {missing}")

    industries = []
    for _, row in df.iterrows():
        tickers = [t.strip() for t in str(row.get("Tickers", "")).split(",") if t.strip()]
        industries.append({
            "rank":          int(row["Rank"])             if pd.notna(row.get("Rank"))             else None,
            "industry":      str(row["Industry"]).strip(),
            "sector":        str(row["Sector"]).strip(),
            "rs":            round(float(row["Relative Strength"]), 2) if pd.notna(row.get("Relative Strength")) else None,
            "percentile":    int(row["Percentile"])       if pd.notna(row.get("Percentile"))       else None,
            "1m_rs_pct":     int(row["1M_RS_Percentile"]) if pd.notna(row.get("1M_RS_Percentile")) else None,
            "3m_rs_pct":     int(row["3M_RS_Percentile"]) if pd.notna(row.get("3M_RS_Percentile")) else None,
            "6m_rs_pct":     int(row["6M_RS_Percentile"]) if pd.notna(row.get("6M_RS_Percentile")) else None,
            "tickers":       tickers,
        })
    print(f"Loaded {len(industries)} industries")
    return industries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",           default="data",                      help="Output directory")
    parser.add_argument("--csv-url",           default=DEFAULT_CSV_URL,             help="URL of RS stock universe CSV")
    parser.add_argument("--industries-csv-url",default=DEFAULT_INDUSTRIES_CSV_URL,  help="URL of RS industries CSV")
    parser.add_argument("--workers",           type=int, default=10,                help="Threads for fallback fetches")
    args = parser.parse_args()

    if not args.csv_url:
        raise SystemExit("Error: --csv-url is required (or set CSV_URL env var)")

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load & filter universe
    universe = load_universe(args.csv_url)
    tickers  = universe["Ticker"].tolist()
    print(f"Universe after filters: {len(tickers)} stocks\n")

    # 2. SPY baseline
    print("Fetching SPY history...")
    spy_hist = yf.Ticker("SPY").history(period="1y").dropna(subset=["Close"])

    # 3. Batch-fetch price histories
    histories = fetch_history_batch(tickers, max_workers=args.workers)

    # 4. Compute metrics and group by industry
    print("Computing metrics...")
    csv_lookup = universe.set_index("Ticker").to_dict(orient="index")

    by_industry:  dict[str, list]  = defaultdict(list)
    industry_to_sector: dict[str, str] = {}
    skipped = 0

    for ticker in tickers:
        hist = histories.get(ticker)
        if hist is None:
            skipped += 1
            continue

        metrics = compute_metrics(ticker, hist, spy_hist)
        if metrics is None:
            skipped += 1
            continue

        csv_row  = csv_lookup.get(ticker, {})
        industry = str(csv_row.get("Industry", "Unknown")).strip() or "Unknown"
        sector   = str(csv_row.get("Sector",   "Unknown")).strip() or "Unknown"

        # Numeric passthrough fields
        passthrough: dict = {}
        for col in CSV_PASSTHROUGH:
            val = csv_row.get(col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                passthrough[col] = None
            else:
                try:
                    passthrough[col] = float(val)
                except (ValueError, TypeError):
                    passthrough[col] = str(val)

        row = {
            "ticker":   ticker,
            "sector":   sector,
            "industry": industry,
            **passthrough,
            **metrics,
        }
        by_industry[industry].append(row)
        industry_to_sector[industry] = sector

    print(f"Done: {len(tickers) - skipped} computed, {skipped} skipped\n")

    # Sort stocks within each industry by Rank (ascending)
    for rows in by_industry.values():
        rows.sort(key=lambda r: float(r.get("Rank") or 9999))

    # 5. Column ranges per industry (for bar-width scaling in UI)
    def vrange(rows, key, d_min, d_max):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return [min(vals), max(vals)] if vals else [d_min, d_max]

    column_ranges = {
        industry: {
            "daily":     vrange(rows, "daily",     -10,  10),
            "1w":        vrange(rows, "1w",         -20,  20),
            "1m":        vrange(rows, "1m",          -30,  30),
            "3m":        vrange(rows, "3m",          -40,  40),
            "ytd":       vrange(rows, "ytd",         -50,  50),
            "vs_spy":    vrange(rows, "vs_spy",      -20,  20),
            "vs_spy_3m": vrange(rows, "vs_spy_3m",   -30,  30),
        }
        for industry, rows in by_industry.items()
    }

    # 6. Industry-level summary (for top-level industry cards / sorting)
    def avg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    def avg_dist_ma50(rows):
        vals = [r["dist_ma"]["SMA50"] for r in rows
                if r.get("dist_ma") and r["dist_ma"].get("SMA50") is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    def median_rs(rows):
        vals = [float(r["Percentile"]) for r in rows if r.get("Percentile") is not None]
        return round(float(np.median(vals)), 1) if vals else None

    industry_summary = {
        industry: {
            "sector":          industry_to_sector[industry],
            "stock_count":     len(rows),
            "median_rs_pct":   median_rs(rows),
            "avg_daily":       avg(rows, "daily"),
            "avg_1w":          avg(rows, "1w"),
            "avg_1m":          avg(rows, "1m"),
            "avg_3m":          avg(rows, "3m"),
            "avg_ytd":         avg(rows, "ytd"),
            "avg_vs_spy":      avg(rows, "vs_spy"),
            "avg_vs_spy_3m":   avg(rows, "vs_spy_3m"),
            "avg_dist_ma50":   avg_dist_ma50(rows),
        }
        for industry, rows in by_industry.items()
    }

    # 7. Sector → industries index
    sector_to_industries: dict[str, list[str]] = defaultdict(list)
    for industry, sector in industry_to_sector.items():
        sector_to_industries[sector].append(industry)
    for sector in sector_to_industries:
        sector_to_industries[sector].sort()

    # 8. Assemble and write outputs
    snapshot = {
        "built_at":         datetime.utcnow().isoformat() + "Z",
        "by_industry":      dict(by_industry),
        "column_ranges":    column_ranges,
        "industry_summary": industry_summary,
    }
    meta = {
        "sector_to_industries": dict(sector_to_industries),
        "industry_to_sector":   industry_to_sector,
        "all_sectors":          sorted(sector_to_industries.keys()),
        "all_industries":       sorted(by_industry.keys()),
        "default_industry":     sorted(by_industry.keys())[0] if by_industry else "",
        "min_avg_volume":       MIN_AVG_VOLUME,
    }

    # Fetch industries if URL provided
    industries_list = []
    if args.industries_csv_url:
        industries_list = load_industries(args.industries_csv_url)
    else:
        print("Warning: --industries-csv-url not set, skipping industries data")

    def write_json(filename, obj):
        path = os.path.join(args.out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sanitize(obj), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        size_kb = os.path.getsize(path) / 1024
        print(f"Wrote {path} ({size_kb:.0f} KB)")

    write_json("snapshot.json", snapshot)
    write_json("meta.json", meta)
    write_json("industries.json", {"built_at": datetime.utcnow().isoformat() + "Z", "industries": industries_list})
    print("\nAll done.")


if __name__ == "__main__":
    main()
