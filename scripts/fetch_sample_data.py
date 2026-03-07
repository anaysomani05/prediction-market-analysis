"""Fetch a sample of real Kalshi data for backtesting validation.

Downloads ~300 resolved markets and their trades across multiple time
windows for diverse category coverage (sports, politics, crypto, finance,
etc.). Saves to data/sample/.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm

from src.indexers.kalshi.client import KalshiClient

SAMPLE_DIR = Path("data/sample")
MARKETS_DIR = SAMPLE_DIR / "markets"
TRADES_DIR = SAMPLE_DIR / "trades"

TARGET_MARKETS = 300  # total number of resolved markets to fetch
MARKETS_PER_WINDOW = 100  # markets per time window
MIN_VOLUME = 50  # minimum trades per market

# Three time windows spread across 2024 for diverse category coverage
TIME_WINDOWS = [
    {"label": "Nov 2023 - Mar 2024", "min_close_ts": 1700000000, "max_close_ts": 1710000000},
    {"label": "Mar 2024 - Jul 2024",  "min_close_ts": 1710000000, "max_close_ts": 1720000000},
    {"label": "Jul 2024 - Oct 2024",  "min_close_ts": 1720000000, "max_close_ts": 1730000000},
]


def fetch_sample():
    MARKETS_DIR.mkdir(parents=True, exist_ok=True)
    TRADES_DIR.mkdir(parents=True, exist_ok=True)

    client = KalshiClient()

    # Fetch resolved markets across multiple time windows
    print(f"Fetching up to {TARGET_MARKETS} resolved markets ({MARKETS_PER_WINDOW} per window, min volume {MIN_VOLUME})...")
    all_markets = []

    for window in TIME_WINDOWS:
        window_markets = []
        cursor = None
        print(f"\n--- Window: {window['label']} ---")

        while len(window_markets) < MARKETS_PER_WINDOW:
            params = {
                "limit": 200,
                "min_close_ts": window["min_close_ts"],
                "max_close_ts": window["max_close_ts"],
            }
            if cursor:
                params["cursor"] = cursor

            try:
                data = client._get("/markets", params=params)
            except Exception as e:
                print(f"API error: {e}")
                break

            markets = data.get("markets", [])
            cursor = data.get("cursor")

            for m in markets:
                if m.get("result") in ("yes", "no") and m.get("volume", 0) >= MIN_VOLUME:
                    window_markets.append(m)
                    if len(window_markets) >= MARKETS_PER_WINDOW:
                        break

            print(f"  Scanned batch, found {len(window_markets)} qualifying markets so far...")

            if not cursor:
                break

        print(f"  Window total: {len(window_markets)} markets")
        all_markets.extend(window_markets)

    if not all_markets:
        print("No markets found. The API may be rate-limited or down.")
        client.close()
        return

    print(f"\nCollected {len(all_markets)} resolved markets across {len(TIME_WINDOWS)} windows. Saving...")

    # Save markets
    markets_df = pd.DataFrame(all_markets)
    markets_df.to_parquet(MARKETS_DIR / "markets.parquet")
    print(f"  Saved {len(markets_df)} markets to {MARKETS_DIR / 'markets.parquet'}")

    # Fetch trades for each market
    print(f"\nFetching trades for {len(all_markets)} markets...")
    all_trades = []
    fetched_at = datetime.utcnow()

    for market in tqdm(all_markets, desc="Fetching trades"):
        ticker = market["ticker"]
        try:
            trades = client.get_market_trades(ticker, verbose=False)
            for t in trades:
                trade_dict = asdict(t)
                trade_dict["_fetched_at"] = fetched_at
                all_trades.append(trade_dict)
        except Exception as e:
            tqdm.write(f"  Error fetching trades for {ticker}: {e}")

    client.close()

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_parquet(TRADES_DIR / "trades.parquet")
        print(f"\nSaved {len(trades_df)} trades to {TRADES_DIR / 'trades.parquet'}")
    else:
        print("No trades fetched.")
        return

    # Summary
    print(f"\n--- Sample Data Summary ---")
    print(f"  Markets: {len(markets_df)}")
    print(f"  Trades:  {len(trades_df)}")
    yes_count = sum(1 for m in all_markets if m["result"] == "yes")
    print(f"  Results: {yes_count} YES / {len(all_markets) - yes_count} NO")
    print(f"  Data saved to: {SAMPLE_DIR.resolve()}")


if __name__ == "__main__":
    fetch_sample()
