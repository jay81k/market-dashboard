"""
Fetch fundamental data for all tickers in snapshot.json via finvizfinance.

Fields fetched per ticker:
  - eps_this_y_pct   : EPS Growth This Year (%)
  - eps_next_y_pct   : EPS Growth Next Year (%)
  - eps_next_5y_pct  : EPS Growth Next 5 Years (%)
  - eps_qoq_pct      : EPS Growth Quarter over Quarter (%)
  - sales_qoq_pct    : Sales Growth Quarter over Quarter (%)
  - profit_margin_pct: Profit Margin (%)

Output: data/fundamentals.json

Run:
  python scripts/fetch_fundamentals.py [--snapshot-path data/snapshot.json] [--out-dir data]

GitHub Actions:
  python scripts/fetch_fundamentals.py --snapshot-path data/snapshot.json --out-dir data
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import requests
from finvizfinance.quote import finvizfinance

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DELAY_BETWEEN_REQUESTS = 0.5   # seconds — keeps Finviz happy
MAX_RETRIES            = 3     # max retry attempts per ticker on transient errors
RETRY_BASE_DELAY       = 2.0   # seconds — exponential backoff base (2, 4, 8)
RETRY_STATUS_CODES     = {429, 502, 503, 504}  # HTTP codes worth retrying

FIELD_MAP = {
    "eps_this_y_pct":    "EPS this Y",
    "eps_next_y_pct":    "EPS next Y Percentage",
    "eps_next_5y_pct":   "EPS next 5Y",
    "eps_qoq_pct":       "EPS Q/Q",
    "sales_qoq_pct":     "Sales Q/Q",
    "profit_margin_pct": "Profit Margin",
    "fwd_pe":            "Forward P/E",
    "ps_ratio":          "P/S",
    "peg_ratio":         "PEG",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_pct(val) -> float | None:
    """Parse a Finviz percentage string like '13.66%' or '-5.20%' into a float."""
    if val is None:
        return None
    try:
        s = str(val).strip().replace("%", "").replace(",", "")
        if s in ("-", "", "N/A"):
            return None
        return round(float(s), 2)
    except (ValueError, TypeError):
        return None


def parse_float(val) -> float | None:
    """Parse a plain Finviz float string like '24.5' or '1.8' into a float."""
    if val is None:
        return None
    try:
        s = str(val).strip().replace(",", "")
        if s in ("-", "", "N/A"):
            return None
        return round(float(s), 2)
    except (ValueError, TypeError):
        return None


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception looks like a transient server error worth retrying."""
    if isinstance(exc, requests.exceptions.HTTPError):
        code = exc.response.status_code if exc.response is not None else None
        return code in RETRY_STATUS_CODES
    # finvizfinance raises plain exceptions with the status text in the message
    msg = str(exc).lower()
    return any(str(c) in msg for c in RETRY_STATUS_CODES)


def fetch_fundamentals(ticker: str) -> dict | None:
    """Fetch and parse fundamental fields for a single ticker, with retry/backoff."""
    last_exc = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            stock     = finvizfinance(ticker)
            fundament = stock.ticker_fundament()
            return {
                "eps_this_y_pct":    parse_pct(fundament.get("EPS this Y")),
                "eps_next_y_pct":    parse_pct(fundament.get("EPS next Y Percentage")),
                "eps_next_5y_pct":   parse_pct(fundament.get("EPS next 5Y")),
                "eps_qoq_pct":       parse_pct(fundament.get("EPS Q/Q")),
                "sales_qoq_pct":     parse_pct(fundament.get("Sales Q/Q")),
                "profit_margin_pct": parse_pct(fundament.get("Profit Margin")),
                "fwd_pe":            parse_float(fundament.get("Forward P/E")),
                "ps_ratio":          parse_float(fundament.get("P/S")),
                "peg_ratio":         parse_float(fundament.get("PEG")),
            }

        except Exception as e:
            last_exc = e
            if _is_retryable(e) and attempt < MAX_RETRIES:
                wait = RETRY_BASE_DELAY * (2 ** attempt)   # 2s, 4s, 8s
                print(f"  RETRY [{ticker}] attempt {attempt + 1}/{MAX_RETRIES} "
                      f"after {wait:.0f}s — {e}")
                time.sleep(wait)
            else:
                # Non-retryable error or retries exhausted
                break

    print(f"  ERROR [{ticker}]: {last_exc}")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-path", default="data/snapshot.json", help="Path to snapshot.json")
    parser.add_argument("--out-dir",       default="data",               help="Output directory")
    args = parser.parse_args()

    # Load tickers from snapshot
    print(f"Loading snapshot: {args.snapshot_path}")
    with open(args.snapshot_path, "r", encoding="utf-8") as f:
        snapshot = json.load(f)

    tickers = []
    for rows in snapshot.get("by_industry", {}).values():
        for row in rows:
            t = row.get("ticker")
            if t:
                tickers.append(t)

    tickers = sorted(set(tickers))
    total   = len(tickers)
    print(f"Tickers to fetch: {total}\n")

    # Fetch fundamentals
    results  = {}
    failed   = []
    start    = time.time()

    for i, ticker in enumerate(tickers, 1):
        data = fetch_fundamentals(ticker)
        if data:
            results[ticker] = data
            print(f"  [{i:04d}/{total}] {ticker:<8} "
                  f"EPS_TY={str(data['eps_this_y_pct']):<8} "
                  f"EPS_NY={str(data['eps_next_y_pct']):<8} "
                  f"EPS_5Y={str(data['eps_next_5y_pct']):<8} "
                  f"EPS_QQ={str(data['eps_qoq_pct']):<8} "
                  f"SLS_QQ={str(data['sales_qoq_pct']):<8} "
                  f"PM={str(data['profit_margin_pct']):<8} "
                  f"FWD_PE={str(data['fwd_pe']):<8} "
                  f"PS={str(data['ps_ratio']):<8} "
                  f"PEG={str(data['peg_ratio'])}")
        else:
            failed.append(ticker)
            print(f"  [{i:04d}/{total}] {ticker:<8} FAILED")

        time.sleep(DELAY_BETWEEN_REQUESTS)

    elapsed = time.time() - start
    print(f"\nDone: {len(results)} succeeded, {len(failed)} failed in {elapsed/60:.1f} min")
    if failed:
        print(f"Failed tickers: {failed}")

    # Write output
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "fundamentals.json")
    output = {
        "built_at":   datetime.utcnow().isoformat() + "Z",
        "ticker_count": len(results),
        "fundamentals": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Wrote {out_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
